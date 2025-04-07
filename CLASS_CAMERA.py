import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

class CameraStreamer:
    def __init__(self, pipeline_str=None):
        """
        __init__

        Constructor

        :param pipeline_str: GStreamer pipeline string. If not given, a default pipeline will be used.

        :ivar pipeline_str: GStreamer pipeline string
        :ivar pipeline: GStreamer pipeline object
        :ivar loop: GStreamer main loop object
        """
        Gst.init(None)  # Initialize GStreamer

        if pipeline_str is None:
            self.pipeline_str = (
                "libcamerasrc ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
                "videoconvert ! "
                "videoscale ! "
                "videoflip method=vertical-flip ! "
                "clockoverlay time-format=\"%D %H:%M:%S\" ! "
                "autovideosink"
            )
        else:
            self.pipeline_str = pipeline_str

        self.pipeline = None
        self.loop = None

    def on_bus_message(self, bus, message, loop):
    """
    Handles messages from the GStreamer bus.

    This method is called whenever a message is received on the GStreamer bus.
    It processes End-of-Stream (EOS) and Error messages. If an EOS message is
    received, the stream ends and the main loop is quit. If an Error message
    is received, the error is logged and the main loop is quit.

    :param bus: The GStreamer bus that received the message.
    :param message: The message received from the bus.
    :param loop: The GStreamer main loop to be quit on EOS or Error.
    :return: Always returns True.
    """

        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            loop.quit()
        return True

    def start_stream(self):
        """
        Starts the camera stream.

        This method starts the camera stream by creating a GStreamer pipeline,
        setting up a main loop to handle messages from the pipeline, and starting
        the pipeline. If an error occurs, it is logged and the stream is stopped.
        If the stream ends, it is also stopped.

        :return: None
        """
        try:
            self.pipeline = Gst.parse_launch(self.pipeline_str)
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            self.loop = GLib.MainLoop()
            bus.connect("message", self.on_bus_message, self.loop)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop.run()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)

    def stop_stream(self):
        """
        Stops the camera stream.

        This method stops the camera stream by setting the GStreamer pipeline
        to the NULL state and quitting the main loop. If the pipeline is not
        running, this method does nothing.

        :return: None
        """
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop and self.loop.is_running():
            self.loop.quit()

if __name__ == "__main__":
    streamer = CameraStreamer()
    streamer.start_stream()