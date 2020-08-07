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
