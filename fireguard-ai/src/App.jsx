import React, { useState, useEffect } from 'react'
import './App.css'
import {useSpring, animated} from "@react-spring/web"

function App() 
{
  const [loading, setLoading] = useState(true);

  const backgroundSpring = useSpring({
    transform: loading ? "translateY(0%)" : "translateY(-100%)",
    config: { tension: 200, friction: 30 },
  })

  const logoSpring = useSpring({
    opacity: loading ? 1 : 0,
    config: { duration: 500 },
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false); // Hide the loading screen after 2 seconds
    }, 1500);

    return () => clearTimeout(timer); // Cleanup timer
  }, []);

  return (
    <div className="App">
      {loading && (
        <animated.div className="loading-screen" style={backgroundSpring}>
          <animated.img
            src="../public/assets/logo.png"
            alt="Logo"
            style={logoSpring}
            className="logo"
          />
        </animated.div>
      )}
    </div>
  );
}


export default App
