import React, { useState, useEffect } from "react";
import { useSpring, animated } from "@react-spring/web";
import "./Loading.css";

function Loading({ onFinish }) {
  const [loading, setLoading] = useState(true);

  const backgroundSpring = useSpring({
    transform: loading ? "translateY(0%)" : "translateY(-100%)",
    config: { tension: 200, friction: 30 },
  });

  const logoSpring = useSpring({
    opacity: loading ? 1 : 0,
    config: { duration: 500 },
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false); // Start the swipe animation
      onFinish(); // Notify parent when loading is done
    }, 1500);

    return () => clearTimeout(timer); // Cleanup timer
  }, [onFinish]);

  return (
    <animated.div className="loading-screen" style={backgroundSpring}>
      <animated.img
        src="../public/assets/logo.png"
        alt="Logo"
        style={logoSpring}
        className="logo"
      />
    </animated.div>
  );
}

export default Loading;
