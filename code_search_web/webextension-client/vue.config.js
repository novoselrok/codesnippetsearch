module.exports = {
    publicPath: '',
    outputDir: 'build/dist',
    indexPath: 'popup.html',
    filenameHashing: false,
    configureWebpack: {
        optimization: {
            splitChunks: false
        }
    },
    css: {
        extract: false
    }
}
