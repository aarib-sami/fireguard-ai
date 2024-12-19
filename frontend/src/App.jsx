import React, { useState } from "react";
import Map from "./components/Map/Map.jsx";
import Search from "./components/SearchBox/SearchBox.jsx";
import "./App.css"; 

function App() {

  return (
    <div>
      <Map />
      <Search/>
    </div>
  );
}

export default App;