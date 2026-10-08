const { chromium } = require('playwright');

(async () => {
  const output = process.argv[2] || 'reports/assets/web_demo.png';
  const imagePath = process.argv[3];
  const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  const browser = await chromium.launch({ executablePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  await page.goto('http://127.0.0.1:7860', { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForTimeout(1500);
  if (imagePath) {
    await page.locator('input[type=file]').first().setInputFiles(imagePath);
    await page.waitForTimeout(1200);
    await page.getByRole('button', { name: '开始检测' }).click();
    await page.getByText('检测结论', { exact: false }).waitFor({ timeout: 180000 });
    await page.waitForTimeout(800);
  }
  await page.screenshot({ path: output, fullPage: true });
  await browser.close();
  console.log(output);
})();
