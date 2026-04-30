const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "production",
  target: "web",
  entry: {
    viewer: path.resolve(__dirname, "src/viewer.js"),
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].bundle.js",
    clean: false,
  },
  resolve: {
    extensions: [".ts", ".js"],
    extensionAlias: {
      ".js": [".js", ".ts"],
    },
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, "src/index.html"),
      filename: "index.html",
      inject: "body",
      scriptLoading: "defer",
    }),
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "src/style.css"),
          to: "style.css",
        },
      ],
    }),
  ],
  devtool: "source-map",
};
