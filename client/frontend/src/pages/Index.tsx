import { useState, useCallback } from "react";
import { Mic, Shield, Zap, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AudioUploader } from "@/components/AudioUploader";
import { WaveformVisualizer } from "@/components/WaveformVisualizer";
import { ProcessingState } from "@/components/ProcessingState";
import { VerificationResult } from "@/components/VerificationResult";
import { IdentificationResult } from "@/components/IdentificationResult";
import { toast } from "@/hooks/use-toast";
import axios from "axios";

type OperationStatus = "idle" | "processing" | "complete";
type AppMode = "verify" | "identify";

interface VerificationResultData {
  prediction: string;
  predicted_id: 0 | 1;
  confidence: number;
}

const API_BASE_URL = "http://10.223.135.81:8000";
const isProcessing = (s: OperationStatus): boolean => s === "processing";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<OperationStatus>("idle");
  const [verificationResultData, setVerificationResultData] =
    useState<VerificationResultData | null>(null);
  const [identificationResult, setIdentificationResult] =
    useState<IdentificationResultType>(null);
  const [confidence, setConfidence] = useState<number | undefined>();
  const [mode, setMode] = useState<AppMode>("verify");

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
    setVerificationResultData(null);
    setIdentificationResult(null);
    setStatus("idle");
  }, []);

  const handleVerify = useCallback(async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please upload an audio file first",
        variant: "destructive",
      });
      return;
    }

    setMode("verify");
    setStatus("processing");
    setVerificationResultData(null);
    setConfidence(undefined);

    const formData = new FormData();
    formData.append("audio_file", selectedFile);

    try {
      const response = await axios.post<VerificationResultData>(`${API_BASE_URL}/verify`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const { prediction, predicted_id, confidence } = response.data;
      setVerificationResultData({ prediction, predicted_id, confidence });
      setConfidence(confidence);
      setStatus("complete");

      toast({
        title: "Verification Complete",
        description: predicted_id === 1
          ? "Voice pattern recognized"
          : "Voice not in whitelist",
      });
    } catch (error: any) {
      console.error("Verification Failed:", error);
      toast({
        title: "Verification Failed",
        description: error.response?.data?.detail || "Could not process the audio file",
        variant: "destructive",
      });
      setStatus("idle");
    }
  }, [selectedFile]);

  const handleIdentify = useCallback(async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please upload an audio file first",
        variant: "destructive",
      });
      return;
    }

    setMode("identify");
    setStatus("processing");
    setIdentificationResult(null);
    setConfidence(undefined); 
    const formData = new FormData();
    formData.append("audio_file", selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/recognize_speaker`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const { name, confidence, authorized } = response.data;
      setIdentificationResult({ name, confidence, authorized });
      setConfidence(confidence);
      setStatus("complete");

      toast({
        title: "Identification Complete",
        description: `Voice identified as ${name}`,
      });
    } catch (error: any) {
      console.error("Identification Failed:", error);
      toast({
        title: "Identification Failed",
        description: error.response?.data?.detail || "Could not process the audio file",
        variant: "destructive",
      });
      setStatus("idle");
    }
  }, [selectedFile]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setVerificationResultData(null);
    setIdentificationResult(null);
    setStatus("idle");
    setConfidence(undefined);
  }, []);

  const renderResult = () => {
    if (status !== "complete") return null;

    if (mode === "verify" && verificationResultData) {
      return (
        <VerificationResult
          prediction={verificationResultData.prediction}
          predicted_id={verificationResultData.predicted_id}
          confidence={verificationResultData.confidence}
        />
      );
    }

    if (mode === "identify" && identificationResult) {
      return (
        <IdentificationResult
          name={identificationResult.name}
          confidence={identificationResult.confidence}
          authorized={identificationResult.authorized}
        />
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen flex flex-col">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/5 rounded-full blur-3xl" />
      </div>

      <main className="relative z-10 flex-1 container mx-auto px-6 py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4 tracking-tight">
              Voice Recognition
            </h2>
            <p className="text-lg text-muted-foreground max-w-lg mx-auto">
              Upload an audio sample to verify against the whitelist or
              identify the speaker.
            </p>
          </div>

          <div className="mb-8 opacity-40">
            <WaveformVisualizer
              isAnimating={status === "processing"}
              barCount={60}
            />
          </div>

          <div className="space-y-6">
            {isProcessing(status) ? (
              <ProcessingState />
            ) : status === "complete" ? (
              <>
                {renderResult()}
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleReset}
                  className="w-full"
                >
                  Start Over
                </Button>
              </>
            ) : (
              <>
                <AudioUploader
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  disabled={isProcessing(status)}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button
                    variant="hero"
                    size="xl"
                    onClick={handleVerify}
                    disabled={!selectedFile || isProcessing(status)}
                    className="w-full"
                  >
                    <Zap className="w-5 h-5" />
                    Verify Identity
                  </Button>
                  <Button
                    variant="hero"
                    size="xl"
                    onClick={handleIdentify}
                    disabled={!selectedFile || isProcessing(status)}
                    className="w-full"
                  >
                    <User className="w-5 h-5" />
                    Identify Speaker
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </main>
      <footer className="relative z-10 border-t border-border/50">
        <div className="container mx-auto px-6 py-4">
          <p className="text-center text-sm text-muted-foreground">
            By Rafał Lasota, Piotr Czechowski, Aleksander Jeżowski, Michał
            Kozicki, Mantas Mikulskis
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;

