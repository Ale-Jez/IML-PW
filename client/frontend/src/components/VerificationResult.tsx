import { CheckCircle, XCircle, Shield } from "lucide-react";
import { cn } from "@/lib/utils";

interface VerificationResultProps {
  status: "authorized" | "unauthorized" | null;
  confidence?: number;
}

export const VerificationResult = ({
  status,
  confidence,
}: VerificationResultProps) => {
  if (!status) return null;

  const isAuthorized = status === "authorized";

  return (
    <div
      className={cn(
        "glass-card p-8 text-center transition-all duration-500 animate-in fade-in slide-in-from-bottom-4",
        isAuthorized ? "shadow-glow-success" : "shadow-glow-destructive"
      )}
    >
      <div
        className={cn(
          "w-24 h-24 rounded-full mx-auto flex items-center justify-center mb-6",
          isAuthorized
            ? "bg-success/20 text-success"
            : "bg-destructive/20 text-destructive"
        )}
      >
        {isAuthorized ? (
          <CheckCircle className="w-14 h-14" />
        ) : (
          <XCircle className="w-14 h-14" />
        )}
      </div>

      <div className="flex items-center justify-center gap-2 mb-2">
        <Shield
          className={cn(
            "w-5 h-5",
            isAuthorized ? "text-success" : "text-destructive"
          )}
        />
        <span
          className={cn(
            "text-sm font-mono uppercase tracking-wider",
            isAuthorized ? "text-success" : "text-destructive"
          )}
        >
          Voice Verification
        </span>
      </div>

      <h2
        className={cn(
          "text-3xl font-bold mb-2",
          isAuthorized ? "text-success" : "text-destructive"
        )}
      >
        {isAuthorized ? "Access Granted" : "Access Denied"}
      </h2>

      <p className="text-muted-foreground">
        {isAuthorized
          ? "Voice pattern matches authorized user"
          : "Voice pattern not found in whitelist"}
      </p>

      {confidence !== undefined && (
        <div className="mt-6 pt-6 border-t border-border">
          <p className="text-sm text-muted-foreground mb-2">Confidence Score</p>
          <div className="flex items-center justify-center gap-3">
            <div className="flex-1 max-w-xs h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-1000",
                  isAuthorized ? "bg-success" : "bg-destructive"
                )}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="font-mono text-lg font-semibold">
                {confidence * 100}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
