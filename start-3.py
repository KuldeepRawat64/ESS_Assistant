from pdf2image import convert_from_path
from PIL import Image

def test_pdf_to_image(pdf_path):
    try:
        pages = convert_from_path(pdf_path)
        for i, page in enumerate(pages):
            page.save(f"page_{i}.png", "PNG")
            print(f"Saved page {i} as PNG")
        return True
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return False

test_pdf_to_image("./data/SRS_ESS_V1.0_Phase1-A.pdf")
