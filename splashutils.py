splash_string = 'Starting Segment Anything Model Image Segmentation Tool (SAMIST)'

try:
    import pyi_splash
except ImportError:
    pyi_splash = None


def splash_update(text: str, splash_str=splash_string):
    if pyi_splash:
        pyi_splash.update_text(f"{splash_string}\n{text}")


def splash_close():
    if pyi_splash:
        pyi_splash.close()
