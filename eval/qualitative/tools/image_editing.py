from PIL import Image

def split_image_in_half(img):
    # Get dimensions
    # Force RGB (drop alpha channel safely)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    width, height = img.size

    # Compute the middle point
    mid = width // 2

    # Split into left and right halves
    left_half = img.crop((0, 0, mid, height))
    right_half = img.crop((mid, 0, width, height))

    return left_half, right_half

def merge_images_horizontally(img1, img2):
    # Ensure both images have the same height
    if img1.height != img2.height:
        raise ValueError("Images must have the same height to merge horizontally.")

    # Create a new blank image with combined width
    total_width = img1.width + img2.width
    merged_image = Image.new('RGB', (total_width, img1.height))

    # Paste the two images side by side
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (img1.width, 0))

    merged_image.show()

    return merged_image