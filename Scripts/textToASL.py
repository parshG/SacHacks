import os
from PIL import Image
from moviepy import ImageSequenceClip


def pad_and_resize_image(image_path, output_path, target_size=(500, 500)):

    img = Image.open(image_path)

    width, height = img.size

    max_side = max(width, height)
    new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))  # White background
    new_img.paste(img, ((max_side - width) // 2, (max_side - height) // 2))  # Center original image

    # Resize to target size while maintaining proportions
    new_img = new_img.resize(target_size, Image.LANCZOS)
    new_img.save(output_path)

def generate_asl_sequence(text, image_folder, output_folder, image_size=(500, 500)):

    text = text.upper().replace("_"," ")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_images = []

    for index, char in enumerate(text):
        if char.isalnum():
            image_path = os.path.join(image_folder, f"{char}.png")
            if os.path.exists(image_path):
                output_path = os.path.join(output_folder, f"{index:02d}_{char}.png")

                pad_and_resize_image(image_path, output_path, image_size)

                saved_images.append(output_path)
            else:
                print(f"Warning: ASL image for '{char}' not found.")
        elif char == " ":
            image_path = os.path.join(image_folder, "blank.png")
            if os.path.exists(image_path):
                output_path = os.path.join(output_folder, f"{index:02d}_{char}.png")

                pad_and_resize_image(image_path, output_path, image_size)

                saved_images.append(output_path)
            else:
                print(f"Warning: ASL image for '{char}' not found.")
        else:
            print(f"Skipping unsupported character: '{char}'")

    print(f"ASL image sequence saved in {output_folder}")
    return saved_images  # Return the list of saved images

def create_blank_image(output_path, size=(500, 500), color=(255, 255, 255)):

    blank = Image.new("RGB", size, color)
    blank.save(output_path)

def create_asl_video(image_list, output_video, frame_rate=1, letter_duration=1):

    if not image_list:
        print("No images to create a video.")
        return

    blank_duration = letter_duration / 2

    # Ensure the blank image exists
    blank_image_path = "blank_image.png"
    if not os.path.exists(blank_image_path):
        create_blank_image(blank_image_path, size=(500, 500))

    frame_sequence = []
    durations = []

    for img_path in image_list:

        frame_sequence.append(img_path)
        durations.append(letter_duration)

        frame_sequence.append(blank_image_path)
        durations.append(blank_duration)

    clip = ImageSequenceClip(frame_sequence, fps=frame_rate, durations=durations)

    # Export the final video
    clip.write_videofile(output_video, codec="libx264", fps=frame_rate)
    print(f"Video saved as {output_video}")


text = "Sample Text"
image_folder = "Assets"
output_folder = "asl_output"
asl_images = generate_asl_sequence(text, image_folder, output_folder, image_size=(500, 500))

output_video = "asl_video.mp4"
create_asl_video(asl_images, output_video, frame_rate=2, letter_duration=2)
