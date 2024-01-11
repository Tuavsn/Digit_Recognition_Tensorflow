import pygame, sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog

# Load pre-trained MNIST model
model = tf.keras.models.load_model('mnist.h5')

# Window dimensions
window_size_x = 1200
window_size_y = 800

# Draw size
draw_size_x = 850
draw_size_y = 800

# Constants
boundary_inc = 5
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
gray = (100, 100, 100)

# Flag for image saving
image_save = False

# Labels for digits
labels = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
          5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

# Initialize pygame
pygame.init()

# Font for displaying text
font = pygame.font.Font('freesansbold.ttf', 20)
font1 = pygame.font.Font('freesansbold.ttf', 25)

# Flag for drawing
is_writing = False

# Store coordinates of the drawn number
number_xcord = []
number_ycord = []

# Counter for image saving
image_cnt = 1

# Flag for prediction
predict = True

# Create the game window
display_surf = pygame.display.set_mode((window_size_x, window_size_y))
pygame.display.set_caption('Digit Board')

def draw_window():
    # Create the drawing surface
    drawing_surface = pygame.Surface((draw_size_x, draw_size_y))
    drawing_surface.fill(black)

    # Draw the drawing surface
    display_surf.blit(drawing_surface, (0, 0))

def select_image_and_predict():
    menu()
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    root.destroy()

    # read image with opencv
    image = cv2.imread(filename)[:,:,0]

    image_rgb = cv2.resize(image, (800,800))

    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    image_pygame = pygame.image.frombuffer(image_rgb.flatten(), image_rgb.shape[:2][::-1], 'RGB')

    display_surf.blit(image_pygame, (0, 0))

    # Convert the image to RGB
    image = np.invert(image)

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

    # Make a prediction using the loaded model
    predict_result = "Predict result: "
    predict_result += str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])
    predict_result_text = font1.render(predict_result, True, red, white)
    display_surf.blit(predict_result_text, (draw_size_x + (window_size_x - draw_size_x) / 2 - 140, 300))
def menu():
    # Create the menu surface
    menu_surface = pygame.Surface((window_size_x - draw_size_x, window_size_y))
    menu_surface.fill(white)

    # Draw the menu surface
    display_surf.blit(menu_surface, (draw_size_x, 0))

    # Draw the group label
    group_label = font.render("Nhóm 8", 1, red)
    display_surf.blit(group_label, (draw_size_x + (window_size_x - draw_size_x)/2 - 25, 60))

    # Draw the projec title
    project_title = font.render("handwritten digit recognition", 1, red)
    display_surf.blit(project_title, (draw_size_x + (window_size_x - draw_size_x)/2 - 140, 100))

    # Draw clear button
    pygame.draw.rect(display_surf, gray, [draw_size_x + (window_size_x - draw_size_x)/2 - 40, 150, 100, 40])
    clear_button_title = font.render("Clear", 1, white)
    display_surf.blit(clear_button_title, (draw_size_x + (window_size_x - draw_size_x) / 2 - 15, 160))

    # Draw import image
    pygame.draw.rect(display_surf, gray, [draw_size_x + (window_size_x - draw_size_x) / 2 - 40, 220, 100, 40])
    clear_button_title = font.render("Import", 1, white)
    display_surf.blit(clear_button_title, (draw_size_x + (window_size_x - draw_size_x) / 2 - 20, 230))

draw_window()
menu()

#Main game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Capture mouse motion while drawing
        if event.type == MOUSEMOTION and is_writing:
            xcord, ycord = event.pos
            if xcord < draw_size_x and ycord < draw_size_y:
                pygame.draw.circle(display_surf, white, (xcord, ycord), 8, 0)
                number_xcord.append(xcord)
                number_ycord.append(ycord)

        # Start drawing on mouse button down
        if event.type == MOUSEBUTTONDOWN:
            # Clear window
            if (draw_size_x + (window_size_x - draw_size_x)/2 - 40 <= event.pos[0] <= draw_size_x + (window_size_x - draw_size_x)/2 - 40 + 100
                    and 150 <= event.pos[1] <= 190):
                draw_window()
            # Import image
            elif (draw_size_x + (window_size_x - draw_size_x)/2 - 40 <= event.pos[0] <= draw_size_x + (window_size_x - draw_size_x)/2 - 40 + 100
                    and 220 <= event.pos[1] <= 260):
                select_image_and_predict()
            else:
                is_writing = True


                # Finish drawing on mouse button release
        if event.type == MOUSEBUTTONUP:
            is_writing = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            if len(number_xcord) > 0 and len(number_ycord) > 0:
                # Define the boundaries of the drawn number
                rect_min_x, rect_max_x = max(number_xcord[0] - boundary_inc, 0), min(window_size_x, number_xcord[-1] + boundary_inc)
                rect_min_y, rect_max_y = max(number_ycord[0] - boundary_inc, 0), min(number_ycord[-1] + boundary_inc, window_size_y)

                number_xcord = []
                number_ycord = []

                # Extract the drawn number as an image array
                img_arr = np.array(pygame.PixelArray(display_surf))[rect_min_x:min(rect_max_x, draw_size_x), rect_min_y:min(rect_max_y, draw_size_y)].T.astype(np.float32)

                # Save the image if enabled
                if image_save:
                    cv2.imwrite(f'image{image_cnt}.png', img_arr)
                    image_cnt += 1

                # Perform prediction if enabled
                if predict:
                    # Preprocess the image for prediction
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255

                    # Make a prediction using the loaded model
                    label = str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                    # Render the predicted label on the screen
                    text_surface = font.render(label, True, red, white)
                    text_rect_obj = text_surface.get_rect()
                    text_rect_obj.left, text_rect_obj.bottom = rect_min_x, rect_max_y
                    display_surf.blit(text_surface, text_rect_obj)

    pygame.display.update()
