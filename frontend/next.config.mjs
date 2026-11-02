// The browser calls /api/..., and Next forwards it to the Python API.
// API_URL is read when the app is built.
const apiUrl = process.env.API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  compress: false, // keeps streamed answers from being held back
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
};

export default nextConfig;
