import React from "react"
import { SearchBox } from '@mapbox/search-js-react';

function Search(props)
{
    
    return(   
        <div className="search-container">
            <SearchBox 
            className = "search-box"
            accessToken={import.meta.env.VITE_MAPBOX_API_KEY}
            options={{
                language: 'en',
            }}
            onRetrieve={(results) => {props.onSearch(results)}}
            onChange={() => {props.onClick()}}
            />
        </div>
    )
}

export default Search;