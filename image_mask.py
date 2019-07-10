import cv2
import numpy as np



def process_image(image_file):
    minBGR = (250, 250, 250)
    maxBGR = (255, 255, 255)
    with open(image_file, 'rb') as f:
        original = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = original
        # imageLAB = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        maskBGR = cv2.inRange(image, minBGR, maxBGR)
        kernel = np.ones((100, 100), np.uint8)
        # dilate maskBGR to take over edges of pixels of color
        maskBGR = cv2.erode(maskBGR, kernel, iterations=2)
        # resultLAB = cv2.bitwise_and(original, original, mask=maskBGR)
        bbox = cv2.boundingRect(maskBGR)
        print(bbox)
        x, y, w, h = bbox
        crop = original[y:y + h, x:x + w]

        # cv2.imshow('result',crop)
        cv2.imwrite(image_file+"_p.png", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        # cv2.imwrite('resultLAB.jpg', resultLAB, [int(cv2.IMWRITE_JPEG_QUALITY), 70])


if __name__ == '__main__':
    image_file = 'Data/-374-rau/-374-rau_out_2.png'





"""
def recognize_remove_mark(doc, image_classif, mark_removed_name=None,
                          dest_path=None, output_folder=None, debug=False):
    original = cv2.imdecode(np.frombuffer(doc.open('.jpg', 'rb').read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    imageLAB = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    maskLAB = cv2.inRange(imageLAB, minLAB, maxLAB)
    kernel = np.ones((5, 5), np.uint8)
    # dilate maskLAB to take over edges of pixels of color
    maskLAB = cv2.dilate(maskLAB, kernel, iterations=2)
    resultLAB = cv2.bitwise_and(original, original, mask=maskLAB)
    bbox = cv2.boundingRect(maskLAB)
    recognized_sign = ''

    if sum(bbox) > 0:
        x, y, w, h = bbox
        original_x = x
        original_y = y
        xc = x + w // 2
        yc = y + h // 2
        # adjust rectangle
        if w < MARK_MIN_SIZE:
            w = MARK_MIN_SIZE
        if h < MARK_MIN_SIZE:
            h = MARK_MIN_SIZE
        if w < h:
            w = h
        elif w > h:
            h = w
        x = xc - w // 2
        if x < 0:
            x = 0
        y = yc - h // 2
        if y < 0:
            y = 0
        if x + w > original.shape[1]:
            w = original.shape[1] - x
        if y + h > original.shape[0]:
            h = original.shape[0] - y

        prefix = ''
        if h > original.shape[0] // 8 or w > original.shape[1] // 8:
            prefix += '_BIG'
        elif h < 10 or w < 10:
            prefix += '_SMALL'

        if h > original.shape[0] // 4 or w > original.shape[1] // 4:
            prefix = '_HUGE'

        if prefix != '_HUGE':
            crop = original[y:y + h, x:x + w]
            crop_resize = cv2.resize(crop, (64, 64))
            crop_resize = crop_resize[..., ::-1]  # BGR to RGB
            crop_batch = crop_resize.reshape((1,) + crop_resize.shape)
            p = image_classif.predict(crop_batch)[0]
            categ = np.argmax(p)
            lookup = ['I', 'O', 'X']
            categ = lookup[categ]
            # print(p, categ)
            recognized_sign = categ
            if mark_removed_name is not None:
                no_sign = original.copy()
                # TODO: fill color as median pixel by luminosity outside mask
                fill_color = original[original_y, original_x]  # take one pixel outside the sign
                maskBinary = maskLAB > 0
                no_sign = cv2.bitwise_and(no_sign, no_sign, mask=cv2.bitwise_not(maskLAB)) + maskBinary[
                    ..., np.newaxis] * fill_color

                if debug:
                    cv2.rectangle(no_sign, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(no_sign, recognized_sign, (x, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 4, 255)

                cv2.imwrite(mark_removed_name, no_sign, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                if output_folder:
                    pp = os.path.join(output_folder, 'classified', str(categ), prefix + doc.id)
                    os.makedirs(os.path.dirname(pp), exist_ok=True)
                    cv2.imwrite(pp + '.jpg', no_sign, [int(cv2.IMWRITE_JPEG_QUALITY), 70 if prefix else 95])

            if dest_path is not None:
                crop_mask = resultLAB[y:y + h, x:x + w]

                # cv2.rectangle(resultSelection, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.imwrite(dest_path + prefix + '.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70 if prefix else 95])
                if not prefix:
                    cv2.imwrite(dest_path + '_MASK' + '.jpg', crop_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            if dest_path is not None:
                cv2.imwrite(dest_path + '_HUGE.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    else:
        # print(f'NOCROP {dest_path}')
        if dest_path is not None:
            cv2.imwrite(dest_path + '_NO_CROP.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    return recognized_sign

"""