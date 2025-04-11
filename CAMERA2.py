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
                "queue ! autovideosink "  # Branch for display
                "t. ! queue ! x264enc tune=zerolatency ! mp4mux ! filesink name=filesink location=recording.mp4 "  # Branch for recording
                "t. ! queue ! x264enc tune=zerolatency bitrate=500 ! rtph264pay ! udpsink host=IP_CUA_MAY_TINH port=5000"  # Branch for streaming
            )
        else:
            self.pipeline_str = pipeline_str

        self.pipeline = None
        self.loop = None
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
        Starts the camera stream and recording.

        This method starts the camera stream by creating a GStreamer pipeline,
        setting up a main loop to handle messages from the pipeline, and starting
        the pipeline. Recording starts automatically with the stream.

        :return: None
        """
        try:
            self.pipeline = Gst.parse_launch(self.pipeline_str)
            self.filesink = self.pipeline.get_by_name("filesink")  # Get the filesink element
            if self.filesink:
                file_name = f"recording_{int(time.time())}.mp4"
                self.filesink.set_property("location", file_name)
                print(f"Started streaming and recording to {file_name}")
            else:
                print("Error: Filesink not found in pipeline")

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            self.loop = GLib.MainLoop()
            bus.connect("message", self.on_bus_message, self.loop)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop.run()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop_stream()  # Ensure proper cleanup

    def stop_stream(self):
        """
        Stops the camera stream and recording.

        This method stops the camera stream by sending an EOS event to the pipeline,
        ensuring the recording is finalized properly, then setting the pipeline to NULL state.

        :return: None
        """
        if self.pipeline:
            # Send EOS to ensure the recording is finalized
            self.pipeline.send_event(Gst.Event.new_eos())
            # Wait for EOS to propagate (give some time for mp4mux to finalize the file)
            time.sleep(1)
            self.pipeline.set_state(Gst.State.NULL)
            print("Stopped streaming and recording")
            if self.filesink:
                file_path = self.filesink.get_property("location")
                print(f"Recording saved to {file_path}")
        if self.loop and self.loop.is_running():
            self.loop.quit()