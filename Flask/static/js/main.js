//modified 
// Store our API endpoint inside queryUrl
//var url = "/static/data/data.csv";
var url = "http://127.0.0.1:5000/api/v1.0/map_data";
url_all = "http://127.0.0.1:5000/api/v1.0/all_accidents";
console.log(url);




d3.json(url_all, function (data) {
    // Once we get a response, send the data.features object to the createFeatures function
    populatetable(data);
});





function somefunction() {
    element = document.getElementById('city');
    console.log(element);
    if (element != null) 
    {
        var chosenValue = d3.select('city').property('value');
        // console.log(chosenValue);
        url = "localhost:5000/submitted/";
        d3.json(url, function(d) 
        {
            createFeatures(d);
            console.log(d);
        })
    }    
    else
    {
        console.log("bad data");
    }
}

function populatetable(filtered)
    {
            // Get a reference to the table body
            var tbody = d3.select("tbody");

            //remove old data
            rows = tbody.selectAll("tr").remove();

            //itereate through filtered data
            filtered.forEach(function(ufoSights) {
                //console.log(ufoSights);
                var row = tbody.append("tr");
                Object.entries(ufoSights).forEach(function([key, value]) 
                {
                    //console.log(key, value);
                    // Append a cell to the row for each value
                    // in the weather report object
                    var cell = row.append("td");
                    cell.text(value);
    			});
    		});

    }


    function handleClick()
    {
        console.log("Clicked");
        event.preventDefault();
        var chosenValue = d3.select('#city').node().value;
        console.log(chosenValue);
        center_map(chosenValue);

    
    
    }
    var button = d3.select("#button");
    button.on("click", handleClick);

