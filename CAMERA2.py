import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import time

class CameraStreamer:
    def __init__(self, pipeline_str=None):
        """
        __init__

        Constructor

        :param pipeline_str: GStreamer pipeline string. If not given, a default pipeline will be used.

        :ivar pipeline_str: GStreamer pipeline string
        :ivar pipeline: GStreamer pipeline object
        :ivar loop: GStreamer main loop object
        :ivar recording: Flag to indicate if recording is active
        """
        Gst.init(None)  # Initialize GStreamer

        if pipeline_str is None:
            self.pipeline_str = (
                "libcamerasrc ! "
                "video/x-raw, width=320, height=240, framerate=24/1 ! "
                "videoconvert ! "
                "videoscale ! "
                "videoflip method=vertical-flip ! "
                "clockoverlay time-format=\"%D %H:%M:%S\" ! "
                "tee name=t ! "
                "queue ! autovideosink "
                "t. ! queue ! x264enc tune=zerolatency ! mp4mux ! filesink location=recording.mp4"
            )
        else:
            self.pipeline_str = pipeline_str

        self.pipeline = None
        self.loop = None
        self.recording = False
        self.filesink = None

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
            self.filesink = self.pipeline.get_by_name("filesink0")  # Get the filesink element
            if self.filesink:
                self.filesink.set_property("location", f"recording_{int(time.time())}.mp4")
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

    def start_recording(self):
        """
        Starts recording the camera stream to a file.

        This method sets the recording flag to True and updates the filesink location
        to a new file with a timestamp in the name.

        :return: None
        """
        if not self.recording and self.pipeline:
            self.recording = True
            if self.filesink:
                self.filesink.set_property("location", f"recording_{int(time.time())}.mp4")
                print(f"Started recording to recording_{int(time.time())}.mp4")
            else:
                print("Error: Filesink not found in pipeline")

    def stop_recording(self):
        """
        Stops recording the camera stream.

        This method sets the recording flag to False and resets the pipeline to start a new file
        for the next recording.

        :return: None
        """
        if self.recording and self.pipeline:
            self.recording = False
            print("Stopped recording")
            # Reset pipeline to prepare for the next recording
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.set_state(Gst.State.PLAYING)

if __name__ == "__main__":
    streamer = CameraStreamer()
    streamer.start_stream()