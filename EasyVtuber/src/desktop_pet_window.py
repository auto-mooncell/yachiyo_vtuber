import tkinter as tk
from PIL import Image, ImageChops, ImageFilter, ImageTk


class DesktopPetWindow:
    """A lightweight borderless window with colorkey transparency on Windows."""

    TRANSPARENT_COLOR = "#00ff00"
    OUTLINE_WIDTH = 4
    OUTLINE_OPACITY = 176
    ALPHA_CUTOFF = 12

    def __init__(self, width, height, title="EasyVtuber Desktop Pet"):
        self.width = width
        self.height = height
        self.closed = False
        self._photo = None
        self._drag_x = 0
        self._drag_y = 0
        self.scale = 1.0

        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)
        self.root.configure(bg=self.TRANSPARENT_COLOR)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", self.TRANSPARENT_COLOR)
        self.root.geometry(f"{width}x{height}+80+80")

        self.label = tk.Label(
            self.root,
            bg=self.TRANSPARENT_COLOR,
            bd=0,
            highlightthickness=0,
        )
        self.label.pack(fill="both", expand=True)

        self.root.bind("<ButtonPress-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)
        self.root.bind("<ButtonPress-3>", self._close)
        self.root.bind("<Escape>", self._close)
        self.root.bind("<MouseWheel>", self._zoom)
        self.label.bind("<ButtonPress-1>", self._start_drag)
        self.label.bind("<B1-Motion>", self._drag)
        self.label.bind("<ButtonPress-3>", self._close)
        self.label.bind("<MouseWheel>", self._zoom)

    def _start_drag(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _drag(self, event):
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _close(self, _event=None):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _zoom(self, event):
        delta = 0
        if getattr(event, "delta", 0) > 0:
            delta = 0.1
        elif getattr(event, "delta", 0) < 0:
            delta = -0.1
        self.scale = max(0.3, min(3.0, self.scale + delta))

    def _compose_with_outline(self, rgba_image):
        alpha = rgba_image.getchannel("A")
        kernel = max(3, self.OUTLINE_WIDTH * 2 + 1)
        dilated = alpha.filter(ImageFilter.MaxFilter(kernel))
        outline_mask = ImageChops.subtract(dilated, alpha)
        if self.OUTLINE_WIDTH > 1:
            outline_mask = outline_mask.filter(ImageFilter.GaussianBlur(radius=self.OUTLINE_WIDTH * 0.35))
        outline_mask = outline_mask.point(
            lambda value: min(255, int(value * (self.OUTLINE_OPACITY / 255.0)))
        )

        transparent_rgba = Image.new("RGBA", rgba_image.size, (255, 255, 255, 0))
        outline = Image.new("RGBA", rgba_image.size, (255, 255, 255, 0))
        outline.putalpha(outline_mask)
        visible_rgba = Image.alpha_composite(transparent_rgba, outline)
        visible_rgba = Image.alpha_composite(visible_rgba, rgba_image)

        # Use binary cutout on the colorkey background to avoid green fringe artifacts.
        binary_mask = visible_rgba.getchannel("A").point(
            lambda value: 255 if value >= self.ALPHA_CUTOFF else 0
        )
        visible_rgb = Image.alpha_composite(
            Image.new("RGBA", rgba_image.size, (255, 255, 255, 255)),
            visible_rgba,
        ).convert("RGB")

        green_background = Image.new("RGB", rgba_image.size, (0, 255, 0))
        green_background.paste(visible_rgb, mask=binary_mask)
        return green_background

    def update(self, rgba_frame):
        if self.closed:
            return False

        rgba_image = Image.fromarray(rgba_frame, mode="RGBA")
        if self.scale != 1.0:
            scaled_size = (
                max(1, int(rgba_image.width * self.scale)),
                max(1, int(rgba_image.height * self.scale)),
            )
            rgba_image = rgba_image.resize(scaled_size, Image.LANCZOS)
        composed = self._compose_with_outline(rgba_image)
        self._photo = ImageTk.PhotoImage(composed)
        self.label.configure(image=self._photo)
        self.root.geometry(f"{composed.width}x{composed.height}+{self.root.winfo_x()}+{self.root.winfo_y()}")

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            return False
        return True
