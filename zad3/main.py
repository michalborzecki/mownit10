import skimage
import skimage.io
import string
import numpy as np
from scipy import signal
from PIL import ImageDraw, ImageFont, Image


def main():
    image = skimage.io.imread("test.jpg", True)
    image = skimage.img_as_float(image)
    inverted_image = 1 - image
    denoise_image(inverted_image, 0.1)
    print("searching for appropriate font size...")
    font_size, font_family = get_font(inverted_image, letters='mkh')
    # font_size, font_family = 34, 'sansserif'
    results = search_for_letters(inverted_image, font_size, font_family, '1839465072dbpqoeasmvxzgkhniljwyctfru?!')
    print("results: ")
    results = get_lines(results, font_size, font_family)
    print('\n'.join(results))


def get_lines(letters_positions, font_size, font_family):
    fonts = load_letters(font_size)[font_family]
    font_shape = fonts['a'].shape
    delta = font_shape[0] / 4
    lines = dict()
    ordered_lines = []
    for letter, positions in letters_positions.items():
        for r, c in positions:
            line = r
            for key in lines.keys():
                if abs(r - key) < delta:
                    line = key
                    break
            if line not in lines.keys():
                lines[line] = []
            lines[line].append((letter, c))
    for line in sorted(lines.keys()):
        ordered_lines.append(sorted(lines[line], key=lambda x: x[1]))
    del lines
    return lines_to_strings(ordered_lines, fonts)


def lines_to_strings(ordered_lines, fonts):
    lines = []
    for ordered_line in ordered_lines:
        line = ''
        for i in range(len(ordered_line) - 1):
            line += ordered_line[i][0]
            between_letters = ordered_line[i+1][1] - ordered_line[i][1]
            next_letter_width = fonts[ordered_line[i+1][0]].shape[1]
            space_delta = fonts['a'].shape[1] / 4
            if between_letters > next_letter_width + space_delta:
                line += ' '
        line += ordered_line[len(ordered_line) - 1][0]
        lines.append(line)
    return lines


def denoise_image(image, level):
    image[image < level*(image.max())] = 0
    image[image >= level*(image.max())] = 1
    return image


def get_text_box(inverted_image, min_margin):
    padding = {'l': 0, 'r': 0, 't': 0, 'b': 0}
    for row in inverted_image:
        if row.sum() == 0:
            padding['t'] += 1
        else:
            break
    for row in reversed(inverted_image):
        if row.sum() == 0:
            padding['b'] += 1
        else:
            break
    for col in range(len(inverted_image[0])):
        if inverted_image[:,col].sum() == 0:
            padding['l'] += 1
        else:
            break
    for col in range(len(inverted_image[0]) - 1, 0, -1):
        if inverted_image[:,col].sum() == 0:
            padding['r'] += 1
        else:
            break
    padding['l'] = max(0, padding['l'] - 10)
    padding['r'] = min(len(inverted_image[0]), padding['r'] - 10)
    padding['t'] = max(0, padding['t'] - min_margin)
    padding['b'] = min(len(inverted_image), padding['b'] - min_margin)
    return {'t': padding['t'], 'b': len(inverted_image) - padding['b'],
            'l': padding['l'], 'r': len(inverted_image[0]) - padding['r']}


def search_for_letters(inverted_image, font_size, font_family, letters):
    fonts = load_letters(font_size)[font_family]
    padding = get_text_box(inverted_image, fonts['a'].shape[0] / 2)
    inverted_image = inverted_image[padding['t']:padding['b'], padding['l']:padding['r']]

    results = dict()
    for i in range(len(letters)):
        letter = letters[i]
        print("searching for " + letter)
        font = fonts[letter]
        font = font / 255
        denoise_image(font, 0.1)
        max_possible_corr = signal.correlate2d(font, font).max()
        corr = signal.correlate2d(inverted_image, font)
        corr = corr / max_possible_corr
        corr[corr < 0.8] = 0
        maxim = get_local_max(corr, font.shape)
        results[letter] = []
        for j in range(len(maxim)):
            r, c = maxim[j]
            source = inverted_image[r - font.shape[0] +1:r+1, c - font.shape[1]+1:c+1]
            cutted = source - font
            cutp = cutted.copy()
            cutp[cutp < 0] = 0
            # print(cutp.sum() / source.sum())
            if np.abs(cutp.sum() / source.sum()) < 0.2:
                inverted_image[r - font.shape[0]:r, c - font.shape[1]:c] = 0
                results[letter].append(maxim[j])
        print("found " + str(len(results[letter])))
    return results


def get_font(inverted_image, min_size=14, max_size=60, letters="ash"):
    fonts = dict()
    for s in range(max_size, min_size, -1):
        fonts[s] = load_letters(s, letters)
    results = {'serif': [-1, 0, 0], 'sansserif': [-1, 0, 0]}
    for font_type in {'serif', 'sansserif'}:
        for letter in letters:
            skip = 0
            first_found = -1
            letter_result = [-1, 0, 0]
            for s in range(max_size, min_size, -1):
                if first_found > 0 and first_found - 3 > s:
                    break
                if skip > 0:
                    skip -= 1
                    continue
                n_font = fonts[s][font_type][letter]
                n_font = n_font / 255
                denoise_image(n_font, 0.1)
                max_possible_corr = signal.correlate2d(n_font, n_font).max()
                corr = signal.correlate2d(inverted_image, n_font)
                corr = corr / max_possible_corr
                corr[corr < 0.8] = 0
                corr_max = corr.max()
                if corr_max < 0.5:
                    skip = 1
                    continue
                maxim = get_local_max(corr, n_font.shape)
                print(font_type + " " + letter + " " + str(s) + ": " + str(len(maxim)) + " found, " + str(corr_max))
                if letter_result[2] < corr_max:
                    if letter_result[0] == -1:
                        first_found = s
                    letter_result = [s, len(maxim), corr_max]
            if letter_result[0] > results[font_type][0]:
                results[font_type] = letter_result[:]
                print("new size")
    if results['serif'][2] > results['sansserif'][2]:
        return results['serif'][0], 'serif'
    else:
        return results['sansserif'][0], 'sansserif'


def load_letters(font_size, letters_to_load=''):
    def load_letter(letter, font, font_height):
        letter_im = Image.new('L', (font.getsize(letter)[0], font_height), 0)
        letter_draw = ImageDraw.Draw(letter_im)
        letter_draw.text((0, 0), letter, font=font, fill=255)
        return np.matrix(np.asarray(letter_im))

    serif_font = ImageFont.truetype('fonts/LiberationSerif-Regular.ttf', font_size)
    serif_font_height = serif_font.getsize("pI")[1]
    sansserif_font = ImageFont.truetype('fonts/LiberationSans-Regular.ttf', font_size)
    sansserif_font_height = sansserif_font.getsize("pI")[1]
    letters = {'serif': dict(), 'sansserif': dict()}
    if len(letters_to_load) == 0:
        letters_to_load = string.ascii_lowercase + string.digits + '.,!?'
    for char in letters_to_load:
        letters['sansserif'][char] = load_letter(char, sansserif_font, sansserif_font_height)
        letters['serif'][char] = load_letter(char, serif_font, serif_font_height)
    return letters

def get_local_max(matrix, block_size):
    max_list = []
    m = matrix.max()
    size = matrix.shape
    while m != 0:
        max_index = np.unravel_index(np.argmax(matrix), size)
        max_list.append(max_index)
        matrix[max(max_index[0]-block_size[0], 0):min(max_index[0]+block_size[0], size[0]),
               max(max_index[1]-block_size[1], 0):min(max_index[1]+block_size[1], size[1])] = 0
        m = matrix.max()
    return max_list

if __name__ == "__main__":
    main()
