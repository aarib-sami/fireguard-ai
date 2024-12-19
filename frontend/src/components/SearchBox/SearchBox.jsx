import React from "react"
import { SearchBox } from '@mapbox/search-js-react';

function Search()
{
    
    return(   
        <div className="search-container">
            <SearchBox 
            className = "search-box"
            accessToken={import.meta.env.VITE_MAPBOX_ACCESS_TOKEN}
            options={{
                language: 'en',
                country: 'CA',
            }}
            />
        </div>
    )
}

export default Search;