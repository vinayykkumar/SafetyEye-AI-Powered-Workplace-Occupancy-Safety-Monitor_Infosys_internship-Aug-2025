import tkinter as tk
from PIL import Image, ImageTk
import cv2
import logging
from queue import Queue, Empty
import time

# Setup logging
logger = logging.getLogger(__name__)

def gui_process(gui_queue):
    root = None
    try:
        root = tk.Tk()
        root.configure(bg='#2E2E2E')
        root.title("Safety Violation Alerts")
        label = tk.Label(root, text="No Violations Detected", font=("Arial", 14), bg='#2E2E2E', fg='white')
        label.pack(pady=10)
        canvas = tk.Label(root, bg='#2E2E2E')
        canvas.pack()

        while True:
            try:
                violations, frame = gui_queue.get(timeout=1.0)
                if violations:
                    violation_text = "\n".join([f"Person not wearing {v[0].replace('NO-', '').replace('Hardhat', 'Hard hat')}: {v[1]:.2f} ({v[2]})" for v in violations])
                    label.config(text=f"Violations Detected:\n{violation_text}", fg="red")
                else:
                    label.config(text="No Violations Detected", fg="white")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                canvas.config(image=photo)
                canvas.image = photo
                root.update()
                gui_queue.task_done()
            except Empty:
                if root and not root.winfo_exists():
                    break
            except Exception as e:
                logger.error(f"GUI update error: {e}")
                time.sleep(0.1)
    except Exception as e:
        logger.error(f"GUI process error: {e}")
    finally:
        if root and root.winfo_exists():
            root.destroy()