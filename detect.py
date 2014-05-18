#!/usr/bin/env python

""""""


import sys
import logging
import os
import time
import cv2


cascade_frontalface = 'alt'
scaleFactor = 1.07  # (1,2] lower means missed faces less likely, non-faces more likely (lower takes longer too)
minNeighbors = 5    # [3,6] lower means missed faces less likely, non-faces more likely


cascades = {
    'frontalface': {
        'default': cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml'),
        'alt': cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    },
    'eye': cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
}


def show_image(image, title=None):
    if title is None:
        title = str(image.shape[1])+'x'+str(image.shape[0])
    cv2.imshow(title, image)
    while True:
        key = cv2.waitKey()
        if key == 27 or key == ord('q'):  # Press esc or q to close.
            break
    cv2.destroyAllWindows()


def process(filename, directory=''):

    """"""

    log = logging.getLogger(__name__)

    log.debug(filename+' Loading...')
    image = cv2.imread(filename)

    log.debug(filename+' Converting to grayscale...')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    log.debug(filename+' Equalizing grayscale histogram...')
    gray = cv2.equalizeHist(gray)

    log.debug(filename+' Detecting faces...')
    faces = cascades['frontalface'][cascade_frontalface].detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )

    log.debug(filename+' Applying rectangles...')
    probabilities = [0, 0, 0, 0, 0]
    for (fx, fy, fw, fh) in faces:
        eyes = len(cascades['eye'].detectMultiScale(gray[fy:fy+fh, fx:fx+fw]))
        probabilities[min(eyes, 4)] += 1
        color = {  # BGR
            1: (0, 255, 255),
            2: (0, 255, 0),
            3: (0, 255, 255)
        }.get(eyes, (0, 0, 255))
        cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, 2)

    log.debug(filename+' Creating a new image...')
    cv2.imwrite(directory+'/'+os.path.basename(filename), image)

    log.info(
        filename+' '+str(len(faces))+' faces (' +
        str(probabilities[2])+' confirmed, ' +
        str(probabilities[1]+probabilities[3])+' probable, ' +
        str(probabilities[0]+probabilities[4])+' potential)'
    )


if __name__ == '__main__':
    log_file = 'detection.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    folder = 'detected/'+str(int(time.time()*1000))+'.'+cascade_frontalface +\
             '.sF-'+str(scaleFactor) +\
             '.mN-'+str(minNeighbors)
    if not os.path.exists(folder):
        os.makedirs(folder)
    arg_max = 0
    for arg in sys.argv[1:]:
        arg_len = len(arg)
        if arg_len > arg_max:
            arg_max = arg_len
    for arg in sys.argv[1:]:
        print('{filename:<'+str(arg_max)+'}').format(filename=arg),
        start = time.time()
        process(arg, folder)
        end = time.time()
        print '{runtime:>15.10f}s'.format(runtime=end-start)
    os.rename(log_file, folder+'/'+log_file)
