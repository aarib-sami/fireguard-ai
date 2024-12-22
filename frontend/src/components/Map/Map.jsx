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
    fetchFireRisk(longitude, latitude);
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
    fetchFireRisk(long, lat);
  }

  function onSearchClick(){
    props.disablePrediction();
  }

  async function fetchFireRisk(longitude, latitude)
  {
    try 
    {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify  ({longitude, latitude}),
      });
      const data = await response.json();
      console.log(data);
      props.enablePrediction(data);
    } catch (error) {
      console.error('Error:', error);
    }
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
