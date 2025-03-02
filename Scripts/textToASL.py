import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from moviepy import ImageSequenceClip
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name = "dummy-cloud-name",
    api_key = "<API_KEY>",
    api_secret = "<API_SECRET>", # Click 'View API Keys' above to copy your API secret
    secure=True
)

app = FastAPI()

# Ensure required directories exist
IMAGE_FOLDER = "Assets"
OUTPUT_FOLDER = "asl_output"
VIDEO_PATH = "asl_video.mp4"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def pad_and_resize_image(image_path, output_path, target_size=(500, 500)):
    """ Adds white padding to make an image square and resizes it. """
    img = Image.open(image_path)
    max_side = max(img.size)
    new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    new_img.paste(img, ((max_side - img.width) // 2, (max_side - img.height) // 2))
    new_img = new_img.resize(target_size, Image.LANCZOS)
    new_img.save(output_path)


def generate_asl_sequence(text, image_folder, output_folder, image_size=(500, 500)):
    """ Generates a sequence of ASL images corresponding to the input text. """
    text = text.upper().replace("_", " ")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_images = []

    for index, char in enumerate(text):
        if char.isalnum():
            image_path = os.path.join(image_folder, f"{char}.png")
        elif char == " ":
            image_path = os.path.join(image_folder, "blank.png")
        else:
            continue  # Skip unsupported characters

        if os.path.exists(image_path):
            output_path = os.path.join(output_folder, f"{index:02d}_{char}.png")
            pad_and_resize_image(image_path, output_path, image_size)
            saved_images.append(output_path)
        else:
            print(f"Warning: ASL image for '{char}' not found.")

    return saved_images


def create_blank_image(output_path, size=(500, 500), color=(255, 255, 255)):
    """ Creates a blank (white) image. """
    blank = Image.new("RGB", size, color)
    blank.save(output_path)


def create_asl_video(image_list, output_video, frame_rate=2, letter_duration=2):
    """ Creates an ASL video from a sequence of images. """
    if not image_list:
        raise HTTPException(status_code=400, detail="No images found to create a video.")

    blank_duration = letter_duration / 2
    blank_image_path = "blank_image.png"

    if not os.path.exists(blank_image_path):
        create_blank_image(blank_image_path, size=(500, 500))

    frame_sequence, durations = [], []
    for img_path in image_list:
        frame_sequence.append(img_path)
        durations.append(letter_duration)
        frame_sequence.append(blank_image_path)
        durations.append(blank_duration)

    clip = ImageSequenceClip(frame_sequence, fps=frame_rate, durations=durations)
    clip.write_videofile(output_video, codec="libx264", fps=frame_rate)
    return output_video


@app.post("/generate-asl-video/")
def generate_video(text: str):
    """ API Endpoint to generate an ASL video from text input. """
    asl_images = generate_asl_sequence(text, IMAGE_FOLDER, OUTPUT_FOLDER, image_size=(500, 500))

    if not asl_images:
        raise HTTPException(status_code=400, detail="Could not generate ASL images.")

    video_file = create_asl_video(asl_images, VIDEO_PATH, frame_rate=2, letter_duration=2)
    if not os.path.exists(video_file):
        raise HTTPException(status_code=500, detail="Video creation failed.")
    response = cloudinary.uploader.upload(
        video_file,
        resource_type="video"
    )

    print(response["secure_url"])  # Get the video URL
    return {"video_url": f"{response['secure_url']}"}


@app.get("/download/{filename}")
def download_video(filename: str):
    """ API Endpoint to download the generated video. """
    file_path = os.path.join(filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename="asl_video.mp4")
    raise HTTPException(status_code=404, detail="File not found.")

@app.get("/")
async def root():
    return {"message": "Hello World"}

