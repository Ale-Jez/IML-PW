import { useState, useRef, useCallback } from "react";
import { Mic, StopCircle, Save, Trash, Waves } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface AudioRecorderProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const AudioRecorder = ({
  onFileSelect,
  disabled,
}: AudioRecorderProps) => {
  const [recordingStatus, setRecordingStatus] = useState<
    "idle" | "recording" | "recorded"
  >("idle");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream; // Store the stream to stop tracks later
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav", // Using WAV as a common format, can be dynamic
        });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        setRecordingStatus("recorded");
        streamRef.current?.getTracks().forEach(track => track.stop()); // Stop microphone access
      };
      audioChunksRef.current = [];
      mediaRecorderRef.current.start();
      setRecordingStatus("recording");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      // Optionally, provide user feedback about microphone access error
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  };

  const saveRecording = () => {
    if (audioUrl) {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      const audioFile = new File([audioBlob], `recording-${Date.now()}.wav`, {
        type: "audio/wav",
      });
      onFileSelect(audioFile);
      discardRecording(); // Clear state after saving
    }
  };

  const discardRecording = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl); // Clean up the object URL
    }
    setRecordingStatus("idle");
    setAudioUrl(null);
    audioChunksRef.current = [];
    streamRef.current?.getTracks().forEach(track => track.stop()); // Ensure microphone access is stopped
  };

  return (
    <div className="flex flex-col items-center justify-center gap-4 text-center">
      <div className="text-center">
        <p className="text-lg font-medium text-foreground">
          Record your voice
        </p>
        <p className="text-sm text-muted-foreground mt-1">
          Click the button below to start recording your audio.
        </p>
      </div>

      {recordingStatus === "idle" && (
        <div
          onClick={disabled ? undefined : startRecording}
          className={cn(
            "glass-card glow-border p-10 cursor-pointer transition-all duration-300 block w-full",
            "flex flex-col items-center justify-center gap-4", // Added for centering content
            "hover:bg-primary/5", // Hover effect
            disabled && "opacity-50 cursor-not-allowed"
          )}
          title="Start recording"
        >
          <div
            className={cn(
              "w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center transition-transform duration-300"
            )}
          >
            <Mic className="w-10 h-10 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              Click to Record
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Start a new audio recording
            </p>
          </div>
        </div>
      )}

      {recordingStatus === "recording" && (
        <div
          onClick={disabled ? undefined : stopRecording}
          className={cn(
            "glass-card glow-border p-10 cursor-pointer transition-all duration-300 block w-full",
            "flex flex-col items-center justify-center gap-4", // Added for centering content
            "hover:bg-destructive/5", // Hover effect for destructive
            disabled && "opacity-50 cursor-not-allowed"
          )}
          title="Stop recording"
        >
          <div
            className={cn(
              "w-20 h-20 rounded-2xl bg-destructive/10 flex items-center justify-center transition-transform duration-300"
            )}
          >
            <StopCircle className="w-10 h-10 text-destructive" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-foreground animate-pulse">
              Recording...
            </p>
            <p className="text-sm text-muted-foreground mt-1 animate-pulse">
              Click to stop
            </p>
          </div>
        </div>
      )}

      {recordingStatus === "recorded" && audioUrl && (
        <div className="glass-card glow-border p-6 w-full flex flex-col gap-4 items-center">
          <audio src={audioUrl} controls className="w-full rounded-md" />
          <div className="flex gap-4">
            <Button onClick={saveRecording} size="lg" title="Save recording">
              <Save className="w-5 h-5 mr-2" />
              Save
            </Button>
            <Button
              onClick={discardRecording}
              variant="outline"
              size="lg"
              title="Discard recording"
            >
              <Trash className="w-5 h-5 mr-2" />
              Discard
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
