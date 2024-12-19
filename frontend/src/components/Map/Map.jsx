import React, { useState } from 'react';
import ReactMapGl from 'react-map-gl';
import "mapbox-gl/dist/mapbox-gl.css";
import './Map.css';
import Button from '@mui/material/Button';
import Search from '../SearchBox/SearchBox';


function Map(props) {
  const [viewport, setViewport] = useState({
    latitude: 54.243459,
    longitude: -9.073448,
    zoom: 1,
  });

  const [moving, setMoving] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  function handleMove(evt)
  {
    setViewport(evt.viewState)
  }

  function handleMapClick(evt) {
    setZoomed(true);
    const { lngLat: { lat: latitude, lng: longitude } } = evt;
    setViewport({
      latitude,
      longitude,
      zoom: 14,
    });
    props.enablePrediction();
  };

  function resetMap(){
    setViewport({
      latitude: 54.243459,
      longitude: -9.073448,
      zoom: 1,
    });
    props.disablePrediction();
  }

  function onSearch(results){
    const {geometry} = results.features[0];
    const [long, lat] = geometry.coordinates;

    setViewport({
      latitude: lat,
      longitude: long,
      zoom: 18,
    })
    props.enablePrediction();
  }

  function onSearchClick(){
    props.disablePrediction();
  }

  return (
    <div className="map-container">
      <ReactMapGl
        {...viewport}
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_API_KEY}
        mapStyle="mapbox://styles/mapbox/satellite-v9"
        onMove={!moving && handleMove}
        onClick={handleMapClick}
        scrollZoom={{ speed: 2 }}
      >
      </ReactMapGl>
      <Button variant="contained" size="small" className="reset-button" onClick={resetMap}>Reset Map</Button>
      <div className="Search">
        <Search onSearch={onSearch} onClick={onSearchClick}/>
      </div >
    </div>
  );
}

export default Map;
