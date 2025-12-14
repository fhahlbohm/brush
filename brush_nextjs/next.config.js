/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: '/htgs_viewer',
  assetPrefix: '/htgs_viewer/',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
