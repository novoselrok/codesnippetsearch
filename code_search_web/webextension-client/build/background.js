var API_URL = 'https://codesnippetsearch.net'

function fetchEndpoint(endpoint) {
    return fetch(`${API_URL}${endpoint}`).then(response => response.json()).catch(err => console.error(err))
}

function postEndpoint(endpoint, data) {
    return fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).then(response => response.json()).catch(err => console.error(err))
}

browser.contextMenus.create({
  id: 'code-selection',
  title: 'Search by code',
  contexts: ['selection']
})

browser.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'code-selection') {
      browser.tabs.sendMessage(tab.id, { type: 'search-by-code', code: info.selectionText })
  }
})

browser.commands.onCommand.addListener(command => {
    if (command === 'toggle-sidebar') {
        browser.tabs.query({ currentWindow: true, active: true }).then((tabs) => {
            const tab = tabs[0]
            browser.tabs.sendMessage(tab.id, { type: 'open-sidebar' })
        })
    }
})

browser.runtime.onMessage.addListener(function (message) {
    if (message.type === 'fetch-endpoint') {
        return fetchEndpoint(message.endpoint)
    } else if (message.type === 'post-endpoint') {
        return postEndpoint(message.endpoint, message.data)
    }
})
