import React from "react"
import "./FireInfo.css"
import CancelIcon from '@mui/icons-material/Cancel';

function FireInfo(props)
{
    return(
        <div className={`fire-info ${props.transitioning ? 'fire-info-slide-out' : ''}`}>
            <CancelIcon className="close-button" onClick={props.disablePrediction}/>
            <h1 className="risk-text">Fire Risk: </h1>
            <h2 className="risk-info">{props.fireRisk.risk}</h2>
            <h2 className="risk-percentage">{props.fireRisk.percentage}%</h2>
            <h1 className="risk-explanation">{props.fireRisk.explanation}</h1>
            <img className= "logo" src="../../../public/assets/Logo.png"/>
        </div>
    )
}

export default FireInfo;