import serpent.cv

HEART_COLORS = {
    "lefts": {
        (255, 0, 0): "RED",
        (64, 64, 0): "yellow",
        (0, 0, 0,): "black"
    }
}

def frame_to_hearts(frame, game):
    heart_positions = range(1, 16)
    heart_labels = [f"HUD_HEART_{position}" for position in heart_positions]

    hearts = list()


    for i, heart_label in enumerate(heart_labels):
        heart = serpent.cv.extract_region_from_image(frame, game.screen_regions[heart_label])
        heart_pixel = heart[0, 0, 0]
        # print(heart_pixel)
        if heart_pixel == 237 or heart_pixel == 0:
            # print('1', heart_label, heart_pixel)
            return -1
        elif heart_pixel != 255:
            # print('2', heart_label, heart_pixel)
            return i
        elif heart_label == 'HUD_HEART_15' and heart_pixel == 255:
            # print('3', heart_label, heart_pixel)
            return 16
        else:
            continue
