export const fetchEndpoint = endpoint => fetch(`${window.API_URL}${endpoint}`).then(response => response.json())
