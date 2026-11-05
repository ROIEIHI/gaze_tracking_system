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

    return [rows[i:i + TEXT_ROWS_PER_PAGE] for i in range(0, len(rows), TEXT_ROWS_PER_PAGE)]


def render_page(rows: List[List[str]], page: int) -> Tuple[np.ndarray, List[Dict]]:
    """Draw one page of rows, returning the BGR image and the position of every word"""
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
                'width': int(word_width),
                'height': TEXT_FONT_SIZE,
                'page': page
            })

            current_x = word_left - space_width

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), word_positions


def _row_top(row_idx: int) -> int:
    """Top of a text row in pixels. The first row starts at the same height on every page"""
    return int(SCREEN_HEIGHT * TEXT_MARGIN_Y) + row_idx * TEXT_LINE_SPACING
