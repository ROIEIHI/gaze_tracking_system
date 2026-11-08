"""
Text layout module for the Gaze Tracking System
Wraps Hebrew reading text into right-aligned rows and pages, and draws them.
Words are placed one at a time from the right margin, so every stored word
position is exactly where that word was drawn. RTL only.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from config import *

# Text has to end above the page number, which starts 93% down the screen
TEXT_BOTTOM_LIMIT = 0.925

# Opening screen instructions, centred under the title
INSTRUCTIONS_TOP = 0.27             # First row, as a share of the screen height
INSTRUCTIONS_BOTTOM_LIMIT = 0.85    # Last row must end above this, leaving room for the position status
INSTRUCTIONS_ROW_HEIGHT = 1.6       # Row height, in multiples of the font size
INSTRUCTIONS_PARAGRAPH_GAP = 0.6    # Extra space between paragraphs, in multiples of the font size


def load_text_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load the first available font from TEXT_FONT_PATHS"""
    for font_path in TEXT_FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    raise ValueError(f"No text font could be loaded from: {TEXT_FONT_PATHS}")


def to_display_order(word: str) -> str:
    """Reorder a word for drawing. The base direction is fixed so '1-5?' lays out like Hebrew text"""
    return get_display(word, base_dir='R')


def paginate_text(text: str) -> List[List[List[str]]]:
    """Wrap text into rows that fit between the margins, then split the rows into pages"""
    font = load_text_font(TEXT_FONT_SIZE)
    max_width = SCREEN_WIDTH - 2 * int(SCREEN_WIDTH * TEXT_MARGIN_X)
    space_width = font.getlength(" ")

    ascent, descent = font.getmetrics()
    text_bottom = _row_top(TEXT_ROWS_PER_PAGE - 1) + ascent + descent
    if text_bottom > int(SCREEN_HEIGHT * TEXT_BOTTOM_LIMIT):
        raise ValueError(
            f"{TEXT_ROWS_PER_PAGE} rows at font {TEXT_FONT_SIZE} and spacing {TEXT_LINE_SPACING} "
            f"end at y={text_bottom}, below the limit of y={int(SCREEN_HEIGHT * TEXT_BOTTOM_LIMIT)}"
        )

    rows = _wrap_rows(text, font, max_width)
    return [rows[i:i + TEXT_ROWS_PER_PAGE] for i in range(0, len(rows), TEXT_ROWS_PER_PAGE)]


def _wrap_rows(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[List[str]]:
    """Fill rows word by word up to max_width pixels"""
    space_width = font.getlength(" ")
    rows = []
    current_row = []
    current_width = 0

    for word in text.split():
        word_width = font.getlength(to_display_order(word))
        if word_width > max_width:
            raise ValueError(f"Word '{word}' is {word_width:.0f} px wide, the text area is {max_width} px")

        row_width = current_width + space_width + word_width if current_row else word_width
        if row_width > max_width:
            rows.append(current_row)
            current_row = [word]
            current_width = word_width
        else:
            current_row.append(word)
            current_width = row_width

    if current_row:
        rows.append(current_row)

    return rows


def layout_instructions(paragraphs: List[str]) -> List[Tuple[str, int]]:
    """Wrap the opening instructions into rows, returning each row's text and top y"""
    font = load_text_font(INSTRUCTION_FONT_SIZE)
    max_width = SCREEN_WIDTH - 2 * int(SCREEN_WIDTH * TEXT_MARGIN_X)
    row_height = int(INSTRUCTION_FONT_SIZE * INSTRUCTIONS_ROW_HEIGHT)
    y = int(SCREEN_HEIGHT * INSTRUCTIONS_TOP)
    rows = []

    for paragraph in paragraphs:
        for row in _wrap_rows(paragraph, font, max_width):
            rows.append((" ".join(row), y))
            y += row_height
        y += int(INSTRUCTION_FONT_SIZE * INSTRUCTIONS_PARAGRAPH_GAP)

    ascent, descent = font.getmetrics()
    bottom = rows[-1][1] + ascent + descent if rows else 0
    if bottom > int(SCREEN_HEIGHT * INSTRUCTIONS_BOTTOM_LIMIT):
        raise ValueError(
            f"The opening instructions end at y={bottom}, below the limit of y={int(SCREEN_HEIGHT * INSTRUCTIONS_BOTTOM_LIMIT)}. "
            f"Shorten them or lower INSTRUCTION_FONT_SIZE"
        )
    return rows


def draw_instructions(image: np.ndarray, paragraphs: List[str]) -> np.ndarray:
    """Draw the opening instructions centred on a BGR image"""
    font = load_text_font(INSTRUCTION_FONT_SIZE)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for text, y in layout_instructions(paragraphs):
        display_text = to_display_order(text)
        draw.text(((SCREEN_WIDTH - font.getlength(display_text)) / 2, y), display_text, font=font, fill=TEXT_COLOR)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def build_reading_sequence() -> List[Dict]:
    """Turn EXPERIMENT into the steps of a session, one step per screen.
    Line and word numbers count through each text, page numbers through the whole session"""
    layout_instructions(OPENING_INSTRUCTIONS)  # Too long to fit stops the session here
    sequence = [{'type': 'message', 'content': OPENING_TITLE, 'details': OPENING_INSTRUCTIONS, 'check_position': True}]
    page_number = 0

    for text in EXPERIMENT:
        part = 0
        first_line = 1
        first_word = 1
        shared = {'text': text['number'], 'text_type': text['type']}

        for kind, content, *correct_answer in text['steps']:
            if kind == 'text':
                part += 1
                for rows in paginate_text(content):
                    page_number += 1
                    sequence.append({
                        **shared, 'type': 'page', 'part': part, 'content': rows, 'page_number': page_number,
                        'first_line': first_line, 'first_word': first_word
                    })
                    first_line += len(rows)
                    first_word += sum(len(row) for row in rows)
            elif kind in ('probe', 'rating'):
                sequence.append({
                    **shared, 'type': kind, 'part': part, 'content': content,
                    'correct_answer': correct_answer[0] if correct_answer else None
                })
            else:
                raise ValueError(f"Text {text['number']} has an unknown step type: {kind}")

    sequence.append({'type': 'message', 'content': CLOSING_MESSAGE})
    return sequence


def render_page(rows: List[List[str]], page: int, first_line: int = 1, first_word: int = 1) -> Tuple[np.ndarray, List[Dict]]:
    """Draw one page of rows, returning the BGR image and the position of every word.
    Line and word numbers carry on from first_line and first_word, so they count through the whole passage"""
    font = load_text_font(TEXT_FONT_SIZE)
    space_width = font.getlength(" ")
    ascent, _ = font.getmetrics()
    letter_height = -font.getbbox("א", anchor="ls")[1]
    right_edge = SCREEN_WIDTH - int(SCREEN_WIDTH * TEXT_MARGIN_X)

    image = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), TEXT_BACKGROUND)
    draw = ImageDraw.Draw(image)
    word_positions = []

    for row_idx, row_words in enumerate(rows):
        baseline = _row_top(row_idx) + ascent
        current_x = right_edge

        for col_idx, word in enumerate(row_words):
            display_word = to_display_order(word)
            word_width = font.getlength(display_word)
            word_left = current_x - word_width

            draw.text((round(word_left), baseline), display_word, font=font, fill=TEXT_COLOR, anchor="ls")

            word_positions.append({
                'word': word,
                'x': int(word_left + word_width / 2),
                'y': baseline - letter_height // 2,
                'row': row_idx,
                'col': col_idx,
                'line': first_line + row_idx,
                'word_index': first_word + len(word_positions),
                'width': int(word_width),
                'height': TEXT_FONT_SIZE,
                'page': page
            })

            current_x = word_left - space_width

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), word_positions


def _row_top(row_idx: int) -> int:
    """Top of a text row in pixels. The first row starts at the same height on every page"""
    return int(SCREEN_HEIGHT * TEXT_MARGIN_Y) + row_idx * TEXT_LINE_SPACING
