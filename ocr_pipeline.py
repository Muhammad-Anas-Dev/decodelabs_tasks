"""
Project 4: Image or Text Recognition (Basic) — Path 1: OCR
DecodeLabs AI Industrial Training Kit

Pipeline:
  1. Load raw image
  2. Pre-process (grayscale -> Gaussian blur -> adaptive/Otsu threshold -> deskew)
  3. Run pytesseract OCR with per-word confidence scores
  4. Filter output by an 80% confidence gate
  5. Display results: printed text + an annotated image with boxes around
     every word that passed the confidence filter
"""

import cv2
import numpy as np
import pytesseract
import argparse
import os

# The Gatekeeper Rule from the project brief: nothing under 80% survives.
CONFIDENCE_THRESHOLD = 80


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return image


def preprocess(image):
    """
    Step 1: Grayscale conversion
        Collapses the 3D RGB matrix into a 1D intensity matrix.
    Step 2: Gaussian blur
        Smooths out noise/artifacts before thresholding.
    Step 3: Adaptive thresholding (Otsu's method)
        Forces every pixel to a binary decision -> clean black/white text.
    Step 4: Deskew
        Detects and corrects any rotation so text sits on a horizontal baseline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's method auto-picks the cutoff intensity (the "88" in the slide deck
    # example is illustrative — Otsu calculates the right value per-image).
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    deskewed = deskew(thresh)
    return deskewed


def deskew(binary_image):
    """Calculates the rotation angle of the text block and rotates it flat."""
    coords = np.column_stack(np.where(binary_image < 255))
    if len(coords) == 0:
        return binary_image

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in a range that needs normalizing
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def run_ocr(processed_image, psm=6):
    """
    Runs pytesseract in 'data' mode so we get a confidence score per word,
    not just a single string. PSM (Page Segmentation Mode) controls how
    Tesseract expects the text to be laid out:
        3  = fully automatic (mixed layouts)
        6  = a single uniform block of text (default here)
        7  = a single line (headers, plates)
        11 = sparse/scattered text (invoices, forms)
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        processed_image, config=config, output_type=pytesseract.Output.DICT
    )
    return data


def filter_and_draw(original_image, ocr_data, threshold=CONFIDENCE_THRESHOLD):
    """
    The 80% Confidence Filter (Gatekeeper Rule):
        if confidence >= threshold: draw_box_and_label()
        else: drop_detection()
    """
    output_image = original_image.copy()
    accepted_words = []

    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(float(ocr_data["conf"][i])) if ocr_data["conf"][i] != "-1" else -1

        if text == "" or conf < threshold:
            continue  # drop_detection()

        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )

        # draw_box_and_label()
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            output_image, f"{conf}%", (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
        )
        accepted_words.append((text, conf))

    return output_image, accepted_words


def main():
    parser = argparse.ArgumentParser(description="Project 4 - OCR Pipeline")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--threshold", type=int, default=CONFIDENCE_THRESHOLD,
                         help="Minimum confidence percentage to accept a word")
    parser.add_argument("--out", default="output_annotated.png",
                         help="Path to save the annotated output image")
    args = parser.parse_args()

    print(f"[1/4] Loading image: {args.image_path}")
    image = load_image(args.image_path)

    print("[2/4] Pre-processing (grayscale -> blur -> Otsu threshold -> deskew)...")
    processed = preprocess(image)
    cv2.imwrite("output_preprocessed.png", processed)

    print(f"[3/4] Running OCR (pytesseract, --psm {args.psm})...")
    ocr_data = run_ocr(processed, psm=args.psm)

    print(f"[4/4] Applying {args.threshold}% confidence gate and drawing results...")
    annotated, accepted = filter_and_draw(image, ocr_data, threshold=args.threshold)
    cv2.imwrite(args.out, annotated)

    print("\n--- RECOGNIZED TEXT (confidence >= {}%) ---".format(args.threshold))
    if not accepted:
        print("(No text passed the confidence threshold. Try a clearer image "
              "or lower --threshold for testing.)")
    else:
        for word, conf in accepted:
            print(f"  {word!r:30s}  confidence: {conf}%")

    full_text = " ".join(w for w, c in accepted)
    print("\n--- FULL STRING ---")
    print(full_text if full_text else "(empty)")

    print(f"\nSaved pre-processed image -> output_preprocessed.png")
    print(f"Saved annotated output    -> {args.out}")


if __name__ == "__main__":
    main()
