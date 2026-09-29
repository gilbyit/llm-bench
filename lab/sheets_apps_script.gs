/**
 * Riceve i risultati del laboratorio (lab/sheets.py) e li scrive nel foglio.
 *
 * Installazione (una volta):
 *  1. Nel Google Sheet: Estensioni > Apps Script, incolla questo file al posto di Codice.gs, salva.
 *  2. Impostazioni progetto (ingranaggio) > Proprietà dello script > aggiungi
 *     LAB_TOKEN = una stringa lunga a caso (la stessa che metterai in LAB_SHEETS_TOKEN).
 *  3. Esegui > seleziona "setup" > Esegui, e concedi i permessi (serve una volta sola).
 *  4. Esegui il deployment > Nuovo deployment > tipo "App web":
 *     Esegui come: Me    Chi può accedere: Chiunque
 *     Copia l'URL che termina con /exec: è LAB_SHEETS_URL.
 *  Se modifichi lo script: Gestisci deployment > modifica > Versione: nuova. L'URL resta uguale.
 *
 * Operazioni (POST JSON con "token"):
 *  {op:"upsert", sheet, key, rows:[{...}]}  aggiorna la riga con la stessa chiave o la aggiunge
 *  {op:"append", sheet, rows:[{...}]}       aggiunge in fondo (Log: tiene le ultime 3000 righe)
 * Le colonne nuove vengono aggiunte da sole, quindi il lato Python può evolvere senza toccare qui.
 */

const LOG_MAX = 3000;
const ORDER = ['Stato', 'Risultati', 'Log'];

function doPost(e) {
  let req;
  try {
    req = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply({ok: false, error: 'JSON non valido'});
  }
  const token = PropertiesService.getScriptProperties().getProperty('LAB_TOKEN');
  if (!token || req.token !== token) return reply({ok: false, error: 'token errato'});

  const lock = LockService.getScriptLock();   // NASGUL e Z87 possono scrivere insieme
  if (!lock.tryLock(30000)) return reply({ok: false, error: 'foglio occupato, riprova'});
  try {
    const rows = req.rows || [];
    if (req.op === 'upsert') upsert(req.sheet, req.key, rows);
    else if (req.op === 'append') append(req.sheet, rows);
    else return reply({ok: false, error: 'op sconosciuta: ' + req.op});
    return reply({ok: true, n: rows.length});
  } catch (err) {
    return reply({ok: false, error: String(err)});
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return reply({ok: true, info: 'app web del laboratorio attiva'});
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function sheetFor(name) {
  const ss = SpreadsheetApp.getActive();
  return ss.getSheetByName(name) || ss.insertSheet(name, Math.max(0, ORDER.indexOf(name)));
}

/** Intestazioni attuali, più quelle nuove presenti nelle righe (aggiunte in fondo). */
function headersFor(sh, rows, key) {
  const lastCol = sh.getLastColumn();
  let head = lastCol ? sh.getRange(1, 1, 1, lastCol).getValues()[0].filter(String) : [];
  const fresh = [];
  rows.forEach(r => Object.keys(r).forEach(k => {
    if (head.indexOf(k) < 0 && fresh.indexOf(k) < 0) fresh.push(k);
  }));
  if (key && head.indexOf(key) < 0 && fresh.indexOf(key) < 0) fresh.unshift(key);
  if (fresh.length) {
    head = head.concat(fresh);
    sh.getRange(1, 1, 1, head.length).setValues([head]).setFontWeight('bold');
    sh.setFrozenRows(1);
  }
  return head;
}

function upsert(name, key, rows) {
  if (!rows.length) return;
  const sh = sheetFor(name);
  const head = headersFor(sh, rows, key);
  const kc = head.indexOf(key);
  const n = sh.getLastRow() - 1;
  const data = n > 0 ? sh.getRange(2, 1, n, head.length).getValues() : [];
  const index = {};
  data.forEach((r, i) => { index[String(r[kc])] = i; });
  rows.forEach(obj => {
    const k = String(obj[key]);
    let i = index[k];
    if (i === undefined) {
      i = data.length;
      data.push(head.map(() => ''));
      index[k] = i;
    }
    head.forEach((h, c) => { if (h in obj) data[i][c] = obj[h]; });
  });
  sh.getRange(2, 1, data.length, head.length).setValues(data);
}

function append(name, rows) {
  if (!rows.length) return;
  const sh = sheetFor(name);
  const head = headersFor(sh, rows, null);
  const vals = rows.map(o => head.map(h => (h in o ? o[h] : '')));
  sh.getRange(sh.getLastRow() + 1, 1, vals.length, head.length).setValues(vals);
  const extra = sh.getLastRow() - 1 - LOG_MAX;
  if (extra > 0) sh.deleteRows(2, extra);
}

/** Da eseguire una volta a mano: crea le schede e chiede i permessi. */
function setup() {
  ORDER.forEach(sheetFor);
  const def = SpreadsheetApp.getActive().getSheetByName('Foglio1') || SpreadsheetApp.getActive().getSheetByName('Sheet1');
  if (def && def.getLastRow() === 0 && SpreadsheetApp.getActive().getSheets().length > 1) {
    SpreadsheetApp.getActive().deleteSheet(def);
  }
}
