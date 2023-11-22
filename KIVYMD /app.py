from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.camera import MDCamera

class CameraApp(MDApp):
    def build(self):
        # Load the KV language file
        Builder.load_string(
            """
BoxLayout:
    orientation: 'vertical'

    MDCamera:
        id: camera
        play: True
"""
        )
        # Return the root widget
        return Builder.load_string

if __name__ == "__main__":
    CameraApp().run()
