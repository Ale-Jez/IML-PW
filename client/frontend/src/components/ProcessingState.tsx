import { WaveformVisualizer } from "./WaveformVisualizer";
import { Loader2 } from "lucide-react";

export const ProcessingState = () => {
  return (
    <div className="glass-card p-8 text-center">
      <div className="relative w-24 h-24 mx-auto mb-6">
        <div className="absolute inset-0 rounded-full bg-primary/20 pulse-glow" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="w-12 h-12 text-primary animate-spin" />
        </div>
      </div>

      <h3 className="text-xl font-semibold text-foreground mb-2">
        Analyzing Voice Pattern
      </h3>
      <p className="text-muted-foreground mb-6">
        Processing audio and matching against whitelist...
      </p>

      <WaveformVisualizer isAnimating={true} className="opacity-60" />

      <div className="mt-6 flex items-center justify-center gap-2 text-sm text-muted-foreground">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        <span className="font-mono">VERIFICATION IN PROGRESS</span>
      </div>
    </div>
  );
};
