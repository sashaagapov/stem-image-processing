const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const REPO = path.resolve(__dirname, "..", "..") + path.sep;
const ASSETS = path.join(REPO, "report", "assets") + path.sep;
const PAGE_W = 12240, PAGE_H = 15840; // US Letter DXA

function heading(text, level) {
  return new Paragraph({ text, heading: level });
}

function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}

function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "DCE6F1" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: !!opts.header, size: 18 })] })],
  });
}

function table(colWidths, rows) {
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((text, ci) => cell(text, { width: colWidths[ci], header: ri === 0 })),
    })),
  });
}

function codeBlock(lines) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    border: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
    },
    spacing: { after: 160 },
    children: lines.flatMap((line, i) => {
      const run = new TextRun({ text: line || " ", font: "Courier New", size: 18 });
      return i === 0 ? [run] : [new TextRun({ text: line || " ", font: "Courier New", size: 18, break: 1 })];
    }),
  });
}

function fullImage(filePath, widthPx, heightPx, caption) {
  const ext = filePath.split(".").pop();
  const img = new ImageRun({ type: ext === "png" ? "png" : "jpg", data: fs.readFileSync(filePath), transformation: { width: widthPx, height: heightPx } });
  const out = [new Paragraph({ children: [img], alignment: AlignmentType.CENTER, spacing: { after: 60 } })];
  if (caption) out.push(new Paragraph({ children: [new TextRun({ text: caption, italics: true, size: 18 })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }));
  return out;
}

function fileCodeBlock(relPath, label) {
  const full = REPO + relPath;
  const lines = fs.readFileSync(full, "utf-8").split("\n");
  const out = [new Paragraph({ children: [new TextRun({ text: label || relPath, bold: true, size: 20 })], spacing: { before: 160, after: 60 } })];
  out.push(codeBlock(lines));
  return out;
}

function imageRow(beforePath, afterPath, beforeLabel, afterLabel) {
  const img = (filePath) => new ImageRun({ type: "png".includes(filePath.split(".").pop()) ? "png" : "jpg", data: fs.readFileSync(filePath), transformation: { width: 200, height: 200 } });
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: [4500, 4500],
    rows: [
      new TableRow({ children: [
        new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph({ children: [img(beforePath)], alignment: AlignmentType.CENTER })] }),
        new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph({ children: [img(afterPath)], alignment: AlignmentType.CENTER })] }),
      ] }),
      new TableRow({ children: [
        new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: beforeLabel, italics: true, size: 18 })], alignment: AlignmentType.CENTER })] }),
        new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: afterLabel, italics: true, size: 18 })], alignment: AlignmentType.CENTER })] }),
      ] }),
    ],
  });
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } },
    },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { font: "Times New Roman", size: 36, bold: true },
        paragraph: { spacing: { before: 0, after: 240 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 28, bold: true },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 24, bold: true },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H } } },
    children: [
      new Paragraph({ text: "Аналіз та обробка супутникових знімків земної поверхні класичними методами", style: "Title" }),
      new Paragraph({ children: [new TextRun({ text: "STEM-проєкт — звіт команди", size: 26 })], alignment: AlignmentType.CENTER, spacing: { after: 80 } }),
      new Paragraph({ children: [new TextRun({ text: "Олександр Агапов, Тимофій Стукало, Микола Ковальський", size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 400 } }),

      heading("1. Анотація та ключові слова", HeadingLevel.HEADING_1),
      p("Проєкт реалізує й порівнює шість пайплайнів класичної (без ML/нейромереж) обробки зображень на вибірці супутникових/аерознімків земної поверхні (датасет UC Merced Land Use). Кожен із трьох учасників команди реалізував по два тристадійні пайплайни: контурний аналіз і геометричні перетворення; колірну корекцію, сегментацію Otsu й статистику; білатеральну фільтрацію, бінарну морфологію й Лапласів контурний аналіз; Фур'є-фільтрацію, текстурний аналіз і підвищення різкості; детектор кутів Харріса, морфологічне стоншення (skeletonization) й регіональне зростання; перетворення Хафа, афінні перетворення й корекцію яскравості/контрасту. Усі стадії реалізовано вручну на numpy, без бібліотек комп'ютерного зору. Якість перевірено вручну порахованими тестами (C3) і на розширеній вибірці зі 100 знімків 6 класів землекористування (критерій D1/C5); показано, зокрема, що метрики PSNR/SSIM коректно застосовні лише до одного з шести пайплайнів, чий фінальний результат лишається порівнянним зображенням, а розширення вибірки з 24 до 100 знімків виявило й усунуло один хибний висновок, зроблений на замалій вибірці."),
      p("Ключові слова: обробка зображень, супутникові знімки, землекористування, Sobel, Harris, Хаф, Otsu, PSNR, SSIM, класичні методи комп'ютерного зору.", { italics: true }),

      heading("2. Вступ", HeadingLevel.HEADING_1),
      p("Автоматичний аналіз супутникових і аерознімків — контурів забудови й доріг, меж земельних ділянок, текстури рослинності й води, геометричної корекції знімків — застосовується в моніторингу землекористування, міському плануванні й екологічних дослідженнях. Мета проєкту — реалізувати повний набір класичних (не ML) методів обробки зображень для цих задач вручну, зрозуміти кожен крок обчислень і кількісно порівняти, наскільки різні типи методів (контурні, порогові, текстурні, геометричні) підходять для різних класів земної поверхні."),
      p("Огляд суміжних робіт (щонайменше 5 джерел, порівняння з обраним підходом) — у docs/related_work.md; коротко: роботи підтверджують, що класичні ознаки (краї, Otsu, текстура GLCM) і далі використовуються в дистанційному зондуванні, а сучасні підходи на згорткових нейромережах (Castelluccio et al., 2015) дають вищу точність класифікації ціною втрати пояснюваності — цей проєкт свідомо обирає прозорість методу над точністю."),

      heading("3. Опис вибірки зображень", HeadingLevel.HEADING_1),
      p("Джерело: UC Merced Land Use Dataset (Yang & Newsam, 2010) — 21 клас землекористування, по 100 знімків 256×256 px, отримані з USGS National Map Urban Area Imagery. Для проєкту відібрано 100 знімків із 6 класів (по 16–17 на клас): agricultural, denseresidential, forest, harbor, parkinglot, river — обрані для різноманітності текстур/контурів/кольору (однорідні поля й ліс, регулярна забудова й розмітка парковки, вода). Вибірку розширено з початкових 24 до 100 знімків (критерій D1 «розширена вибірка», бонус C5) — розділ 6 показує, що це не формальність: на малій вибірці один із висновків виявився статистичним шумом. Зображення — у data/samples/ (100 файлів, ~2,5 МБ), зведені метрики по кожному з 600 прогонів (100 зображень × 6 пайплайнів) — у results/metrics_summary.csv (не в репозиторії, відтворюється командою python -m common.compare)."),

      heading("4. Опис методів/пайплайнів і обґрунтування параметрів", HeadingLevel.HEADING_1),
      heading("4.1. member1 — пайплайни 1–2", HeadingLevel.HEADING_2),
      p("Пайплайн 1: Sobel (контурний аналіз) → вирівнювання гістограми → поворот/масштабування. Поріг Sobel (threshold=100) підібрано емпірично на тестовій вибірці — досить високий, щоб відсікти дрібний шум текстури, досить низький, щоб не втратити реальні межі об'єктів. Кут повороту (15°) і масштаб (1.0) — параметри геометричної корекції знімка за потреби вирівнювання до карти."),
      p("Пайплайн 2: баланс білого (gray-world) + перетворення RGB→HSV з підвищенням насиченості (SATURATION_BOOST=1.2, щоб кольори на знімку були виразнішими без спотворення відтінку) → сегментація Otsu (поріг підбирається автоматично максимізацією міжкласової дисперсії, без ручного параметра) → статистики (середнє/дисперсія/ентропія) яскравості."),
      heading("4.2. member2 — пайплайни 3–4", HeadingLevel.HEADING_2),
      p("Пайплайн 3: білатеральний фільтр (sigma_spatial=2 — вікно ширше за зерно шуму й вужче за дорогу; sigma_range=50 — шум з σ≈25 згладжується, але межі з різницею кольору понад 100, напр. дах/поле, не змішуються) → бінарна ерозія з диском радіуса 1 (прибирає лише 1–2-піксельне сміття, будівлі від ~6 px лишаються) → Laplacian (8-зв'язний, реагує й на діагональні межі)."),
      p("Пайплайн 4: низькочастотна гаусова FFT-фільтрація (cutoff=40 — для 256×256 лишає структури від ~6 px) → текстурна статистика GLCM-подібна (вікно 9×9, 16 рівнів яскравості) → unsharp mask (amount=1.0, sigma=2.0) застосований до текстурної карти, а не до оригіналу — тому фінальний результат структурний, не фотореалістичний."),
      heading("4.3. member3 — пайплайни 5–6", HeadingLevel.HEADING_2),
      p("Пайплайн 5: детектор кутів Харріса (threshold=0.01 від максимуму відгуку, harris_k=0.04 — стандартне емпіричне значення з оригінальної роботи Харріса–Стівенса) → морфологічне стоншення (skeletonization, алгоритм Zhang–Suen, max_iterations=100) → регіональне зростання (seed_threshold=20, 4-зв'язність) для сегментації однорідних ділянок. Початково стадія 2 була відкриттям/закриттям, але ерозія/дилатація з квадратним вікном — бінарні аналоги фільтрів мінімуму/максимуму, вже реалізованих member3 у ЛР6, тож метод замінено на стоншення (інший метод тієї самої категорії «Морфологічна обробка»), щоб усунути перетин за вимогами методички."),
      p("Пайплайн 6: перетворення Хафа для виявлення ліній (edge_threshold=100, 180 кроків по куту) → афінне перетворення (параметри — поворот/масштаб/зсув — застосовуються за результатом виявлених ліній) → корекція яскравості/контрасту (contrast=1.1) — єдиний пайплайн, чий фінальний результат лишається звичайним порівнянним зображенням (детальніше — розділ 6)."),

      heading("5. Лістинг ключових функцій", HeadingLevel.HEADING_1),
      p("Повний код усіх 18 стадій і 61 тест — у репозиторії: https://github.com/sashaagapov/stem-image-processing (гілка main, src/memberN/pipelineM/, tests/memberN/). Нижче — по одній представницькій функції від кожного учасника."),
      p("member1 — обчислення градієнта Собеля (stage_1_sobel.py):"),
      codeBlock([
        "def sobel_magnitude(image: Image.Image) -> np.ndarray:",
        "    array = np.asarray(image.convert(\"RGB\"), dtype=np.float64)",
        "    gray = _to_grayscale(array)",
        "    height, width = gray.shape",
        "    padded = np.pad(gray, 1, mode=\"reflect\")",
        "    gx = _convolve3x3(padded, _GX, height, width)",
        "    gy = _convolve3x3(padded, _GY, height, width)",
        "    return np.sqrt(gx ** 2 + gy ** 2)",
      ]),
      p("member2 — вага сусіда в білатеральному фільтрі (stage_1_bilateral_filter.py):"),
      codeBlock([
        "spatial = math.exp(-(dx * dx + dy * dy) / (2 * sigma_spatial ** 2))",
        "color_diff_sq = np.sum((neighbour - rgb) ** 2, axis=-1)",
        "range_weight = np.exp(-color_diff_sq / (2 * sigma_range ** 2))",
        "weight = spatial * range_weight",
        "weighted_sum += weight[..., None] * neighbour",
        "weight_total += weight",
      ]),
      p("member3 — обчислення градієнтів для структурного тензора Харріса (stage_1_harris_corners.py):"),
      codeBlock([
        "gray = _gray_array(image)",
        "padded = np.pad(gray, 1, mode=\"edge\")",
        "gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]",
        "gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]",
      ]),

      heading("6. Результати", HeadingLevel.HEADING_1),
      p("Повна таблиця прогонів (600 = 6 пайплайнів × 100 зображень) — results/metrics_summary.csv. Нижче — приклади «до/після» і зведена таблиця середніх PSNR/SSIM по класах для єдиного пайплайну, де ці метрики коректно застосовні за методологією C1 (TZ.md) — пайплайн 6 (member3), фінал якого лишається звичайним зображенням:"),
      imageRow(ASSETS + "harbor_before.jpg", ASSETS + "harbor_after_pipeline1.png", "harbor_01.jpg — оригінал", "Пайплайн 1: Sobel → вирівнювання → поворот"),
      p(""),
      imageRow(ASSETS + "denseresidential_before.jpg", ASSETS + "denseresidential_after_pipeline5.png", "denseresidential_01.jpg — оригінал", "Пайплайн 5: Харріс → стоншення → регіональне зростання"),
      p(""),
      table([2800, 1800, 1800, 1200], [
        ["Клас", "PSNR, дБ", "SSIM", "n"],
        ["agricultural", "30.63", "0.991", "17"],
        ["denseresidential", "27.42", "0.965", "17"],
        ["forest", "35.46", "0.922", "17"],
        ["harbor", "26.87", "0.949", "17"],
        ["parkinglot", "27.41", "0.975", "16"],
        ["river", "31.42", "0.986", "16"],
      ]),
      p("Поріг з TZ.md (PSNR ≥ 20 дБ, SSIM ≥ 0,7) виконано з великим запасом на всіх 6 класах. На початковій вибірці (n=4 на клас) forest мав найнижчий SSIM (0,819) — на розширеній вибірці (n=16–17) цей ефект зник: forest тепер найвищий (0,922), а найнижчий — harbor (0,949, теж далеко вище порогу). Це показує, що висновок на малій вибірці був статистичним шумом, а не реальною закономірністю (детальніше — розділ 7). Повний аналіз і порівняння всіх 6 пайплайнів — report/assets/pipeline_comparison.md.", { italics: true }),
      ...fullImage(ASSETS + "chart_pipeline6_psnr.png", 450, 257, "Рис. 1. Пайплайн 6: PSNR по класах"),
      ...fullImage(ASSETS + "chart_pipeline6_ssim.png", 450, 257, "Рис. 2. Пайплайн 6: SSIM по класах"),
      ...fullImage(ASSETS + "chart_structural_psnr.png", 500, 268, "Рис. 3. Структурні пайплайни 1–5: PSNR відносно оригіналу по класах (індикатор зміни, не якості)"),

      heading("6.1. Особисте порівняння пайплайнів (варіант 2: пайплайн №1 із пайплайном №2)", HeadingLevel.HEADING_2),
      p("member1: пайплайн 1 (Sobel, бінарна карта контурів) проти пайплайну 2 (Otsu-маска+статистика). Середній PSNR 6,7 дБ проти 8,3 дБ, середній SSIM 0,08 проти 0,29 — пайплайн 2 структурно ближчий до оригіналу, бо Otsu-маска зберігає суцільні області того самого розміру/форми, що й вихідні об'єкти, тоді як розріджена карта контурів Sobel — майже випадковий відносно оригіналу патерн білих ліній. Обидва лишаються в структурній групі C1: жоден не призначений бути «схожим на фото»."),
      p("member2: пайплайн 3 (bilateral→morph→Laplacian, розріджена карта контурів) проти пайплайну 4 (FFT→texture→unsharp, текстурна карта). PSNR вищий у пайплайну 4 (10,5 проти 8,5 дБ), але SSIM — навпаки, вищий у пайплайну 3 (0,125 проти 0,092): наочний приклад того, що PSNR і SSIM можуть «не погоджуватись» — PSNR пайплайну 4 підвищує здебільшого forest (15,1 дБ), тоді як SSIM пайплайну 3 рівномірніший по класах."),
      p("member3: пайплайн 5 (сегментаційна маска) проти пайплайну 6 (порівнянне фото) — найконтрастніша пара в усьому проєкті. SSIM пайплайну 5 — 0,04 (типово для структурного виходу), пайплайну 6 — 0,97 (типово для фотореалістичного виходу). Це той самий контраст, що обґрунтовує поділ усіх 6 пайплайнів на дві групи C1 у TZ.md, тепер видно на даних одного учасника."),

      heading("7. Обговорення", HeadingLevel.HEADING_1),
      p("Для пайплайнів зі структурним фіналом (1–5) PSNR/SSIM відносно оригіналу — не оцінка якості, а лише індикатор того, наскільки сильно пайплайн змінив зображення (докладний розбір і конкретні причини — у pipeline_comparison.md). Показово: на класі forest пайплайн 5 (Харріс → стоншення → регіональне зростання) дає найвищий PSNR серед класів (11,1 дБ), хоча результат — переважно темна карта з тонким скелетом, що структурно майже не несе інформації; збіг пояснюється темним кольором каналу лісу, а не якістю обробки. Це наочний приклад того, чому PSNR виключено з критерію C1 для структурних пайплайнів."),
      p("Обмеження методів: Sobel і Laplacian чутливі до дрібної текстурної зашумленості (ліс дає багато хибних контурів); Otsu й регіональне зростання передбачають відносно бімодальний/контрастний розподіл яскравості і дають вироджений результат на однорідних ділянках; детектор Харріса й перетворення Хафа налаштовані на чіткі кути/прямі лінії — на органічних формах (крони дерев, річка) дають слабший і менш стабільний відгук, ніж на регулярній забудові чи розмітці парковки."),
      p("Окреме спостереження методологічного характеру: розширення вибірки з 24 до 100 зображень (розділ 6) показало, що один із висновків пайплайну 6 на малій вибірці (forest — найгірший клас за SSIM) був статистичним шумом і зник на більшій вибірці, тоді як напрямок ефектів у пайплайнах 1, 3, 5 (forest — викид в обидва боки через темний колір каналу) підтвердився на вчетверо більшій вибірці. Це означає, що для метрик по класах земної поверхні мала вибірка (n=4) недостатня для надійних висновків, а n≈17 — уже дає стійкий результат."),

      heading("8. Висновки", HeadingLevel.HEADING_1),
      p("Реалізовано й перевірено шість класичних пайплайнів обробки зображень на розширеній вибірці зі 100 супутникових знімків 6 класів землекористування (критерій D1/C5). Підтверджено емпірично (а не лише теоретично), що єдина кількісна метрика якості за критерієм C1 застосовна лише до пайплайну з фінальним порівнянним зображенням (пайплайн 6); для решти якість варто оцінювати через еталонні сценарії з наперед відомою відповіддю, а не через PSNR на реальних фото. Розширення вибірки також показало практичну цінність критерію D1 поза формальним бонусом: воно виявило й відкинуло хибний висновок, зроблений на малій вибірці. Напрямки подальшої роботи: кількісне порівняння з CNN-підходом (Castelluccio et al., 2015) на тому самому датасеті, адаптивний підбір seed_threshold/harris_k залежно від класу земної поверхні, розширення вибірки на усі 21 клас UC Merced."),

      heading("9. Список джерел (APA)", HeadingLevel.HEADING_1),
      p("Castelluccio, M., Poggi, G., Sansone, C., & Verdoliva, L. (2015). Land use classification in remote sensing images by convolutional neural networks. arXiv:1508.00092."),
      p("Katiyar, S. K., & Arun, P. V. (2014). Comparative analysis of common edge detection techniques in context of object extraction. arXiv:1405.6132."),
      p("Mishra, V. N., Rai, P. K., Prasad, R., Punia, M., & Nistor, M.-M. (2019). Impact of GLCM textural features on classification accuracy using multi-sensor remote sensing data. Earth Science Informatics, 12(1), 71–86. https://doi.org/10.1007/s12145-018-0369-z"),
      p("Srinivas, M., et al. (2019). Remote sensing image segmentation using OTSU algorithm. International Journal of Computer Applications, 178(12), 1–5."),
      p("Yang, Y., & Newsam, S. (2010). Bag-of-visual-words and spatial extensions for land-use classification. In Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems (pp. 270–279). ACM."),

      new Paragraph({ children: [new PageBreak()] }),
      heading("Додаток. Повний код усіх 18 стадій", HeadingLevel.HEADING_1),
      p("Той самий код, що й у репозиторії (https://github.com/sashaagapov/stem-image-processing, src/), станом на момент формування звіту."),
      heading("member1 — пайплайн 1", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member1/pipeline1/stage_1_sobel.py"),
      ...fileCodeBlock("src/member1/pipeline1/stage_2_histogram_equalization.py"),
      ...fileCodeBlock("src/member1/pipeline1/stage_3_geometric_transform.py"),
      heading("member1 — пайплайн 2", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member1/pipeline2/stage_1_color_correction.py"),
      ...fileCodeBlock("src/member1/pipeline2/stage_2_otsu_segmentation.py"),
      ...fileCodeBlock("src/member1/pipeline2/stage_3_statistics.py"),
      heading("member2 — пайплайн 1", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member2/pipeline1/stage_1_bilateral_filter.py"),
      ...fileCodeBlock("src/member2/pipeline1/stage_2_morphology.py"),
      ...fileCodeBlock("src/member2/pipeline1/stage_3_laplacian.py"),
      heading("member2 — пайплайн 2", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member2/pipeline2/stage_1_fourier_filter.py"),
      ...fileCodeBlock("src/member2/pipeline2/stage_2_texture_analysis.py"),
      ...fileCodeBlock("src/member2/pipeline2/stage_3_unsharp_mask.py"),
      heading("member3 — пайплайн 1", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member3/pipeline1/stage_1_harris_corners.py"),
      ...fileCodeBlock("src/member3/pipeline1/stage_2_morphology_open_close.py"),
      ...fileCodeBlock("src/member3/pipeline1/stage_3_region_growing.py"),
      heading("member3 — пайплайн 2", HeadingLevel.HEADING_2),
      ...fileCodeBlock("src/member3/pipeline2/stage_1_hough_transform.py"),
      ...fileCodeBlock("src/member3/pipeline2/stage_2_affine_transform.py"),
      ...fileCodeBlock("src/member3/pipeline2/stage_3_brightness_contrast.py"),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath = path.join(REPO, "report", "report.docx");
  fs.writeFileSync(outPath, buf);
  console.log("written", buf.length, "bytes ->", outPath);
});
