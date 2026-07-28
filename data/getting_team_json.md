## Getting Your `team.json` File

This guide explains how to get a `data/team.json` file when using `"team_data": "json"`.

### Do you actually need this?

Most users should just use `"team_data": "id"` with your `team_id` set - it pulls your squad from the FPL API and calculates all necessary information. You should only use `"team_data": "json"` (or `"json_string"`) if:
- you have already made transfers this gameweek, or
- you have already been using `"team_data": "json"` and there have been price changes relevant to your team since you last updated it

### Steps

You'll create a "bookmarklet" once — a bookmark whose address is a snippet of JavaScript instead of a URL. Tapping/clicking it while on the FPL site runs the snippet and gives you your team data. This works the same way on desktop and mobile browsers.

The snippet:

```javascript
javascript:(async()=>{try{const me=await fetch("https://fantasy.premierleague.com/api/me/"),md=await me.json(),id=md.player.entry,teamResp=await fetch(`https://fantasy.premierleague.com/api/my-team/${id}/`),teamData=await teamResp.json(),json=JSON.stringify(teamData);try{await navigator.clipboard.writeText(json);alert("Your team data has been copied to the clipboard.")}catch(err){prompt("Copy your team JSON below:",json)}}catch(e){alert(e)}})();
```

**Desktop (Chrome / Firefox / Edge / Safari)**

1. Make sure your bookmarks bar is visible (usually `Ctrl+Shift+B` / `Cmd+Shift+B`).
2. Bookmark any page (e.g. `Ctrl+D` / `Cmd+D`) so you have something to edit.
3. Right-click the new bookmark → **Edit**, rename it to something like `Get Team JSON`, and replace its URL with the snippet above. Save.
4. Log in to [fantasy.premierleague.com](https://fantasy.premierleague.com) and go to any page within the site (e.g. your Points or My Team page).
5. Click the `Get Team JSON` bookmark.

**Mobile (iOS Safari / Android Chrome)**

1. Log in to [fantasy.premierleague.com](https://fantasy.premierleague.com) in your browser.
2. Bookmark the page (Safari: Share → Add Bookmark; Chrome: ⋮ menu → Star/Add bookmark).
3. Open your bookmarks list, find the new bookmark, and edit it (Safari: swipe left → Edit; Chrome: ⋮ next to the bookmark → Edit). Rename it to `Get Team JSON` and replace its URL with the snippet above. Save.
4. Navigate back to a page on fantasy.premierleague.com, then open the bookmark again from your bookmarks list — this time it'll run the snippet instead of navigating away.

**After running it**

You'll either see an alert saying the data was copied to your clipboard, or a popup box containing the JSON text (if clipboard access was blocked) — select all and copy it in that case.

Paste the copied JSON into a new file at `data/team.json` in this repository, and save it. It should look like a large JSON object starting with `{"picks": [...`.

Set `"team_data": "json"` in `data/user_settings.json` and run the solver as normal.

### Troubleshooting

- **"Authentication credentials were not provided"** if you try to visit `https://fantasy.premierleague.com/api/my-team/<id>/` directly in the browser — this is expected. The API needs an auth token that your browser only attaches when the request is made *from* the FPL site itself (as the bookmarklet does), not from a bare URL visit.
- **Nothing happens / alert shows an error** — make sure you're logged in and on a page within `fantasy.premierleague.com` (not a search result or a cached page) when you tap/click the bookmark.
- **The bookmark just navigates to a blank/broken page instead of running the script** — the `javascript:` part of the URL was likely stripped or altered when saving. Re-edit the bookmark and paste the snippet again, making sure the URL field starts with `javascript:`.
- This JSON reflects your team at the moment you ran the snippet. If you make transfers on the FPL site afterwards, run it again to get an up-to-date `team.json`.

### A note on privacy

The JSON contains your squad, bank balance, and chip usage - not a password or anything that grants account access - but it's still the latest data about your team. Only share it if you don't mind people knowing what transfers you have made before the deadline.
