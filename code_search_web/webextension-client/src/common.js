export const fetchEndpoint = endpoint => fetch(`${window.API_URL}${endpoint}`).then(response => response.json()).catch(err => console.error(err))
export const postEndpoint = (endpoint, data) =>
    fetch(`${window.API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).then(response => response.json()).catch(err => console.error(err))
