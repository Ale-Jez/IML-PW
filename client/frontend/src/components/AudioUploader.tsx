import { useCallback, useState } from "react";
import { Upload, FileAudio, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { AudioRecorder } from "./AudioRecorder";

interface AudioUploaderProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
}

export const AudioUploader = ({
  onFileSelect,
  selectedFile,
  disabled = false,
}: AudioUploaderProps) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (disabled) return;

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("audio/")) {
        onFileSelect(file);
      }
    },
    [onFileSelect, disabled]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const clearFile = useCallback(() => {
    onFileSelect(null);
  }, [onFileSelect]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (selectedFile) {
    return (
      <div className="glass-card glow-border p-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center">
            <FileAudio className="w-7 h-7 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-medium text-foreground truncate">
              {selectedFile.name}
            </p>
            <p className="text-sm text-muted-foreground font-mono">
              {formatFileSize(selectedFile.size)} • {selectedFile.type.split("/")[1]?.toUpperCase()}
            </p>
          </div>
          <button
            onClick={clearFile}
            disabled={disabled}
            className="w-10 h-10 rounded-lg bg-muted hover:bg-destructive/20 hover:text-destructive transition-all flex items-center justify-center disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <label
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "glass-card glow-border p-10 cursor-pointer transition-all duration-300 block",
          isDragOver && "border-primary/50 bg-primary/5",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileInput}
          disabled={disabled}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-4">
          <div
            className={cn(
              "w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center transition-transform duration-300",
              isDragOver && "scale-110"
            )}
          >
            <Upload className="w-10 h-10 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              Drop your audio file here
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse • WAV, MP3, M4A, FLAC supported
            </p>
          </div>
        </div>
      </label>
      <div className="flex items-center gap-4 my-4">
        <div className="flex-1 h-px bg-border" />
        <p className="text-muted-foreground font-medium">OR</p>
        <div className="flex-1 h-px bg-border" />
      </div>
      <AudioRecorder onFileSelect={onFileSelect} disabled={disabled} />
    </div>
  );
};
