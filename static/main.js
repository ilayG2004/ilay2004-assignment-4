document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();

    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            displayResults(data);
            displayChart(data);
        });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}

//Destroy & remake canvas when displayChart is ran again
var similarityChart;
function displayChart(data) {
    // Input: data (object) - contains the following keys:
    //        - documents (list) - list of documents
    //        - indices (list) - list of indices   
    //        - similarities (list) - list of similarities
    // TODO: Implement function to display chart here
    //       There is a canvas element in the HTML file with the id 'similarity-chart'
    documents = data.documents
    indices = data.indices
    similarities = data.similarities

    var xValues = indices
    var yValues = similarities
    var barColors = "blue";

    if (similarityChart) {
        similarityChart.destroy(); //Create new instance so that the canvas is cleaned
    }

    similarityChart = new Chart("similarity-chart", {
        type: 'bar',
        data: {
            labels: xValues,
            datasets: [{
                backgroundColor: barColors,
                data: yValues
            }]
        },
        options: {
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: "Cosine Similarity"
                }
            }
        }
    });
}