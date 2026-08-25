const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');

// Simple HTTP server to serve the web files
const server = http.createServer((req, res) => {
  let filePath = path.join(__dirname, req.url === '/' ? 'test-runner.html' : req.url);
  
  // Security: prevent directory traversal
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
  };
  
  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end('Not Found');
      } else {
        res.writeHead(500);
        res.end('Server Error');
      }
      return;
    }
    
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
    res.end(content);
  });
});

async function runTests() {
  return new Promise((resolve) => {
    server.listen(8080, async () => {
      console.log('🌐 Test server running on http://localhost:8080');
      
      const browser = await chromium.launch({ headless: true });
      const page = await browser.newPage();
      
      // Capture console logs
      page.on('console', msg => {
        const text = msg.text();
        if (text.includes('✓') || text.includes('✅') || text.includes('❌') || 
            text.includes('---') || text.includes('===') || text.includes('Passed') || 
            text.includes('Failed') || text.includes('Test failed') || text.includes('Running')) {
          console.log(text);
        }
      });
      
      page.on('pageerror', error => {
        console.error('❌ Page error:', error.message);
      });
      
      try {
        console.log('\n📄 Loading test page...');
        await page.goto('http://localhost:8080', { waitUntil: 'networkidle', timeout: 30000 });
        
        // Wait for app.js to load and initialize
        await page.waitForFunction(() => typeof runAllUITests === 'function', { timeout: 10000 });
        console.log('✅ App loaded, UI tests available');
        
        // Run the UI tests
        console.log('\n🧪 Running UI tests...\n');
        const result = await page.evaluate(() => {
          return runAllUITests();
        });
        
        await browser.close();
        server.close();
        resolve(result);
      } catch (error) {
        console.error('❌ Test execution failed:', error.message);
        await browser.close();
        server.close();
        resolve(false);
      }
    });
  });
}

runTests().then(success => {
  console.log('\n' + '='.repeat(50));
  if (success) {
    console.log('🎉 ALL UI TESTS PASSED!');
    process.exit(0);
  } else {
    console.log('💥 SOME UI TESTS FAILED!');
    process.exit(1);
  }
}).catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});