import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


Gst.init(None)

def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, Debug: {debug}")
        loop.quit()
    return True


pipeline_str = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "videoscale ! "
    "videoflip method=vertical-flip ! "  
    "clockoverlay time-format=\"%D %H:%M:%S\" ! "
    "autovideosink"
)


pipeline = Gst.parse_launch(pipeline_str)


bus = pipeline.get_bus()
bus.add_signal_watch()
loop = GLib.MainLoop()


bus.connect("message", on_bus_message, loop)


pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()
except Exception as e:
    print(f"Error: {e}")
finally:
    pipeline.set_state(Gst.State.NULL)