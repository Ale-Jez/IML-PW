import { useEffect, useState } from "react";

interface WaveformVisualizerProps {
  isAnimating?: boolean;
  barCount?: number;
  className?: string;
}

export const WaveformVisualizer = ({
  isAnimating = false,
  barCount = 40,
  className = "",
}: WaveformVisualizerProps) => {
  const [bars, setBars] = useState<number[]>([]);

  useEffect(() => {
    const generateBars = () => {
      return Array.from({ length: barCount }, () => 
        isAnimating ? Math.random() * 100 : 20 + Math.random() * 30
      );
    };

    setBars(generateBars());

    if (isAnimating) {
      const interval = setInterval(() => {
        setBars(generateBars());
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isAnimating, barCount]);

  return (
    <div className={`flex items-center justify-center gap-[2px] h-16 ${className}`}>
      {bars.map((height, index) => (
        <div
          key={index}
          className="w-1 bg-primary/60 rounded-full transition-all duration-100"
          style={{
            height: `${height}%`,
            animationDelay: `${index * 0.02}s`,
          }}
        />
      ))}
    </div>
  );
};
