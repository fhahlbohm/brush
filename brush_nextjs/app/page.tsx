'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense, lazy } from 'react';

const BrushViewer = lazy(() => import('./components/BrushViewer'));

function Loading() {
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      fontSize: '18px',
    }}>
      Loading Brush WASM...
    </div>
  );
}

function Brush() {
  const searchParams = useSearchParams();
  const url = searchParams.get('url');
  // This mode used to be called "zen" mode, keep it for backwards compatibility.
  const fullsplat = searchParams.get('fullsplat')?.toLowerCase() == 'true' || searchParams.get('zen')?.toLowerCase() == 'true' || false;
  return <BrushViewer url={url} fullsplat={fullsplat} />;
}

export default function Home() {

  return (
    <Suspense fallback={<Loading />}>
      <Brush />
    </Suspense>
  );
}
