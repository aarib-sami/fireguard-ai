import React, { useState } from "react";
import Map from "./components/Map/Map.jsx";
import Search from "./components/SearchBox/SearchBox.jsx";
import FireInfo from "./components/FireInfo/FireInfo.jsx";
import "./App.css"; 

function App() {
  
  const [needPrediction, setNeedPrediction] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const [fireRisk, setFireRisk] = useState({risk: "", percentage: ""});

  function disablePrediction()
  {
    setTransitioning(true);
    setTimeout(() => {
      setNeedPrediction(false);
      setTransitioning(false);
    }, 500); 
    
  }
  
  function enablePrediction(fireRisk)
  {
    setNeedPrediction(true);
    setFireRisk({risk: fireRisk.risk, percentage: fireRisk.percentage, explanation: fireRisk.explanation});
  }
  return (
    <div className="App">
      <div className="banner">
        ⚠️ This is just a demo of the functionality. Live weather API is not in use. Results will not be accurate. Backend will spin down, results may take 30+ seconds to show.
      </div>
      <div className="Map">
        <Map 
        needPrediction={needPrediction}
        enablePrediction={enablePrediction}
        disablePrediction={disablePrediction}
        />
      </div>
      {needPrediction && <div>
        <FireInfo 
        transitioning={transitioning}
        disablePrediction={disablePrediction}
        fireRisk={fireRisk}
        />
      </div>
      }
    </div>
  );
}

export default App;