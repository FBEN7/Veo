const { createApp } = Vue;

createApp({
    data() {
        return {
            matches: [],
            reports: {},
            selectedMatchId: null,
            page: 'overview',
            loading: true,
            loadError: '',
            clubTeam: '',
            isDemo: false,
            demoNotice: '',
            role: localStorage.getItem('club-role') || 'staff',
            accountName: localStorage.getItem('club-name') || '',
            accountEmail: localStorage.getItem('club-email') || '',
            accountOpen: false,
            signInOpen: false,
            signInName: '',
            signInEmail: '',
            signInRole: 'staff',
            selectedPlayerId: null,
            positionLens: 'midfielder',
            playerSearch: '',
            assistantQuestion: '',
            assistantBusy: false,
            assistantMessages: [],
            mobileNavOpen: false,
            clubhouse: { club_name: 'Riverside Athletic', fixtures: [], results: [], table: [], uploads_enabled: false },
            clubhouseKey: sessionStorage.getItem('clubhouse-write-key') || '',
            clubhouseKeyVerified: false,
            clubNameDraft: '',
            fixtureDraft: { match_date: '', kickoff: '', opponent: '', venue: '', home: true },
            resultDraft: { match_date: '', home_team: '', away_team: '', home_score: '', away_score: '', venue: '', reported_by: '' },
            clubhouseBusy: false,
            clubhouseMessage: '',
            dataSource: localStorage.getItem('dashboard-data-source') || 'demo',
            rosterEditorOpen: false,
            rosterSaving: false,
            rosterDraft: [],
            reportColumns: [
                { key: 'goals', label: 'GOALS', active: true },
                { key: 'assists', label: 'ASSISTS', active: true },
                { key: 'n_shots', label: 'SHOTS', active: true },
                { key: 'rating', label: 'RATING', active: true },
                { key: 'distance_m', label: 'DISTANCE', active: false },
                { key: 'top_speed_kmh', label: 'TOP SPEED', active: false },
                { key: 'passes_completed', label: 'PASSES', active: false },
                { key: 'tackles', label: 'TACKLES', active: false },
                { key: 'interceptions', label: 'INTERCEPTIONS', active: false },
                { key: 'xg', label: 'xG', active: false },
            ],
            squadColumns: [
                { key: 'goals', label: 'GOALS', active: true },
                { key: 'assists', label: 'ASSISTS', active: true },
                { key: 'n_shots', label: 'SHOTS', active: true },
                { key: 'rating', label: 'RATING', active: true },
                { key: 'passes_completed', label: 'PASSES', active: false },
                { key: 'tackles', label: 'TACKLES', active: false },
                { key: 'interceptions', label: 'INTERCEPTIONS', active: false },
                { key: 'distance_m', label: 'DISTANCE', active: false },
                { key: 'top_speed_kmh', label: 'TOP SPEED', active: false },
            ],
        };
    },
    computed: {
        currentReport() {
            return this.selectedMatchId ? this.reports[this.selectedMatchId] : null;
        },
        currentGoals() {
            return (this.currentReport?.goals || [])
                .filter((goal) => this.normalizeTeam(goal.team) === this.normalizeTeam(this.clubTeam))
                .sort((a, b) => Number(a.timestamp_s) - Number(b.timestamp_s));
        },
        matchRows() {
            return this.matches.map((match) => {
                const report = this.reports[match.id];
                return { ...match, report, result: report ? this.resultFor(report) : null };
            });
        },
        clubMatches() {
            return this.matchRows.filter((match) => match.report);
        },
        sortedMatches() {
            return [...this.matchRows].sort((a, b) => (b.date || '').localeCompare(a.date || ''));
        },
        recentMatches() {
            return this.sortedMatches.slice(0, 5);
        },
        season() {
            const latestDate = this.sortedMatches.find((match) => match.date)?.date;
            const date = latestDate ? new Date(`${latestDate}T00:00:00`) : new Date();
            const startYear = date.getMonth() >= 6 ? date.getFullYear() : date.getFullYear() - 1;
            return `${startYear}/${String(startYear + 1).slice(-2)}`;
        },
        seasonResults() {
            return this.clubMatches.map((match) => ({ ...match, result: this.resultFor(match.report) }));
        },
        wins() { return this.seasonResults.filter((match) => match.result?.outcome === 'W').length; },
        draws() { return this.seasonResults.filter((match) => match.result?.outcome === 'D').length; },
        losses() { return this.seasonResults.filter((match) => match.result?.outcome === 'L').length; },
        winRate() {
            const decided = this.wins + this.draws + this.losses;
            return decided ? Math.round((this.wins / decided) * 100) : null;
        },
        seasonTotals() {
            const reports = this.clubMatches.map((match) => match.report);
            const clubRows = reports.map((report) => this.clubTeamStats(report)).filter(Boolean);
            const players = reports.flatMap((report) => report.players || []);
            const goals = clubRows.reduce((sum, team) => sum + (Number(team.n_goals) || 0), 0);
            const distance = clubRows.reduce((sum, team) => sum + (Number(team.total_distance_km) || 0), 0);
            const xg = clubRows.reduce((sum, team) => sum + (Number(team.xg) || 0), 0);
            const possessionValues = clubRows.map((team) => this.percentValue(team.possession_pct)).filter(Number.isFinite);
            const passes = clubRows.reduce((sum, team) => sum + (Number(team.n_passes) || 0), 0);
            const passesAttempted = clubRows.reduce((sum, team) => sum + (Number(team.passes_attempted) || 0), 0);
            const passesCompleted = clubRows.reduce((sum, team) => sum + (Number(team.passes_completed) || Number(team.n_passes) || 0), 0);
            const tackles = players.reduce((sum, player) => sum + (Number(player.tackles) || 0), 0);
            const assists = players.reduce((sum, player) => sum + (Number(player.assists) || 0), 0);
            const average = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
            return {
                games: this.matches.length,
                goals,
                xg,
                distance,
                averagePossession: average(possessionValues),
                passCompletion: passesAttempted ? passesCompleted / passesAttempted * 100 : null,
                passes,
                tackles,
                assists,
                players: new Set(players.map((player) => player.track_id)).size,
            };
        },
        recentForm() {
            return this.seasonResults.slice(0, 5).map((match) => match.result?.outcome || '?');
        },
        seasonPulse() {
            return this.seasonResults.slice().reverse().map((match) => ({
                id: match.id,
                label: match.label || `Match ${match.id}`,
                outcome: match.result?.outcome || '?',
                goals: Number(match.result?.for) || 0,
                against: Number(match.result?.against) || 0,
            }));
        },
        teamTrend() {
            return this.seasonResults.slice().reverse().map((match, index) => {
                const team = this.clubTeamStats(match.report) || {};
                return {
                    id: match.id,
                    index: index + 1,
                    label: match.label || `Match ${match.id}`,
                    outcome: match.result?.outcome || '?',
                    goals: Number(team.n_goals) || Number(match.result?.for) || 0,
                    conceded: Number(match.result?.against) || 0,
                    possession: this.percentValue(team.possession_pct),
                    passes: Number(team.n_passes) || 0,
                    distance: Number(team.total_distance_km) || 0,
                };
            });
        },
        seasonLeaders() {
            const players = this.squad;
            const top = (key) => [...players].sort((a, b) => (Number(b[key]) || 0) - (Number(a[key]) || 0))[0] || null;
            return {
                distance: top('distance_m'),
                goals: top('goals'),
                passes: top('passes_completed'),
            };
        },
        squad() {
            const appearances = new Map();
            this.sortedMatches.forEach((match) => {
                (match.report?.players || []).forEach((player) => {
                    const key = player.track_id;
                    if (!appearances.has(key)) appearances.set(key, []);
                    appearances.get(key).push(player);
                });
            });
            return [...appearances.values()].map((rows) => {
                const latest = rows[0];
                const sum = (key) => rows.reduce((total, player) => total + (Number(player[key]) || 0), 0);
                const attempts = sum('passes_attempted');
                const completed = sum('passes_completed');
                return {
                    ...latest,
                    appearances: rows.length,
                    minutes_tracked: sum('minutes_tracked'),
                    distance_m: sum('distance_m'),
                    top_speed_kmh: Number((sum('top_speed_kmh') / rows.length).toFixed(1)),
                    n_sprints: sum('n_sprints'),
                    n_passes: completed || sum('n_passes'),
                    passes_completed: completed || sum('n_passes'),
                    passes_attempted: attempts,
                    pass_completion_rate: attempts ? completed / attempts : latest.pass_completion_rate,
                    n_shots: sum('n_shots'),
                    n_goals: sum('n_goals'),
                    goals: sum('goals'),
                    assists: sum('assists'),
                    xg: Number(sum('xg').toFixed(2)),
                    possession_pct: rows.reduce((total, player) => total + (Number(player.possession_pct) || 0), 0) / rows.length,
                    tackles: sum('tackles'),
                    interceptions: sum('interceptions'),
                    carries: sum('carries'),
                    carry_distance_m: sum('carry_distance_m'),
                    rating: Number((sum('rating') / rows.length).toFixed(1)),
                };
            }).sort((a, b) => a.jersey_number - b.jersey_number);
        },
        filteredSquad() {
            const query = this.playerSearch.trim().toLowerCase();
            if (!query) return this.squad;
            return this.squad.filter((player) => `${player.name} ${player.position} ${player.preferred_position} ${player.team || ''} ${player.jersey_number}`.toLowerCase().includes(query));
        },
        selectedPlayer() {
            const accountPlayer = this.role === 'player' && this.accountName
                ? this.squad.find((player) => String(player.name || '').toLowerCase() === this.accountName.toLowerCase())
                : null;
            return accountPlayer || this.squad.find((player) => String(player.track_id) === String(this.selectedPlayerId)) || this.squad[0] || null;
        },
        playerMetrics() {
            const metricSets = {
                defender: [
                    { label: 'Distance covered', key: 'distance_m', suffix: 'km', divisor: 1000, decimals: 2 },
                        { label: 'Tackles', key: 'tackles', suffix: '', decimals: 0 },
                        { label: 'Pass completion', key: 'pass_completion_rate', suffix: '%', multiplier: 100, decimals: 1 },
                    { label: 'Possession', key: 'possession_pct', suffix: '%', decimals: 1 },
                ],
                midfielder: [
                    { label: 'Distance covered', key: 'distance_m', suffix: 'km', divisor: 1000, decimals: 2 },
                        { label: 'Passes completed', key: 'passes_completed', suffix: '', decimals: 0 },
                    { label: 'Possession', key: 'possession_pct', suffix: '%', decimals: 1 },
                        { label: 'Assists', key: 'assists', suffix: '', decimals: 0 },
                ],
                forward: [
                        { label: 'Goals', key: 'goals', suffix: '', decimals: 0 },
                    { label: 'Shots', key: 'n_shots', suffix: '', decimals: 0 },
                        { label: 'Expected goals', key: 'xg', suffix: 'xG', decimals: 2 },
                    { label: 'Top speed', key: 'top_speed_kmh', suffix: 'km/h', decimals: 1 },
                ],
            };
            const player = this.selectedPlayer;
            return (metricSets[this.positionLens] || metricSets.midfielder).map((metric) => {
                const value = player ? Number(player[metric.key]) : NaN;
                const peerValues = this.squad.map((item) => Number(item[metric.key])).filter(Number.isFinite);
                const average = peerValues.length ? peerValues.reduce((sum, item) => sum + item, 0) / peerValues.length : null;
                const scale = metric.key === 'possession_pct' && Number.isFinite(value) && value <= 1 ? 100 : 1;
                const adjustedValue = Number.isFinite(value) ? value * scale * (metric.multiplier || 1) / (metric.divisor || 1) : null;
                const adjustedAverage = average === null ? null : average * (metric.key === 'possession_pct' && average <= 1 ? 100 : 1) * (metric.multiplier || 1) / (metric.divisor || 1);
                return { ...metric, value: adjustedValue, average: adjustedAverage, percent: adjustedAverage ? Math.min(100, Math.round((adjustedValue / adjustedAverage) * 50)) : 0 };
            });
        },
        displayName() {
            return this.accountName || (this.role === 'player' ? 'Player account' : 'Club staff');
        },
        clubName() {
            return this.clubhouse.club_name || 'Riverside Athletic';
        },
        clubInitials() {
            return this.clubName.split(/\s+/).filter(Boolean).slice(0, 2).map((word) => word[0].toUpperCase()).join('') || 'FC';
        },
        standings() {
            return this.clubhouse.table || [];
        },
        pageTitle() {
            return ({ overview: 'Season overview', matches: 'Match centre', squad: 'Squad performance', report: 'Match report', fixtures: 'Fixtures', league: 'League table', club: 'Club setup' })[this.page] || 'Season overview';
        },
    },
    methods: {
        async requestJson(url, options = {}) {
            const response = await fetch(url, options);
            const value = await response.json();
            if (!response.ok || value?.error) throw new Error(value?.error || `Request failed (${response.status})`);
            return value;
        },
        async loadWorkspace() {
            this.loading = true;
            this.loadError = '';
            try {
                const [season, clubhouse] = await Promise.all([
                    this.requestJson(`/api/season?source=${this.dataSource}`),
                    this.requestJson('/api/clubhouse'),
                ]);
                const matches = season.matches;
                if (!Array.isArray(matches) || !season.reports) throw new Error('Could not load the sample season');
                this.clubhouse = clubhouse;
                this.clubNameDraft = clubhouse.club_name;
                this.resultDraft.reported_by ||= clubhouse.club_name;
                if (!clubhouse.uploads_enabled) this.clubhouseKeyVerified = false;
                this.matches = matches;
                this.reports = season.reports;
                this.isDemo = Boolean(season.demo);
                this.demoNotice = season.data_notice || '';
                this.clubTeam = season.club_team || '';
                const firstTeam = this.reports[matches[0]?.id]?.team_stats?.[0]?.team;
                if (!this.clubTeam && firstTeam) this.clubTeam = firstTeam;
                if (!this.selectedPlayerId && this.squad.length) this.selectedPlayerId = this.squad[0].track_id;
            } catch (error) {
                this.loadError = error.message;
            } finally {
                this.loading = false;
            }
        },
        async verifyClubhouseKey() {
            this.clubhouseMessage = '';
            try {
                await this.requestJson('/api/clubhouse/verify', {
                    method: 'POST',
                    headers: { 'X-Clubhouse-Key': this.clubhouseKey },
                });
                this.clubhouseKeyVerified = true;
                sessionStorage.setItem('clubhouse-write-key', this.clubhouseKey);
                this.clubhouseMessage = 'Key verified. Club edits and result entry are enabled for this browser session.';
            } catch (error) {
                this.clubhouseKeyVerified = false;
                this.clubhouseMessage = error.message;
            }
        },
        async saveClubName() {
            const previousName = this.clubName;
            await this.writeClubhouse('/api/clubhouse/club', { club_name: this.clubNameDraft }, 'PUT', 'Club name updated.', true);
            if (this.clubName === this.clubNameDraft.trim() && previousName !== this.clubName) {
                if (!this.resultDraft.reported_by || this.resultDraft.reported_by === previousName) this.resultDraft.reported_by = this.clubName;
            }
        },
        async createFixture() {
            const opponent = this.fixtureDraft.opponent.trim();
            const club = this.clubName;
            await this.writeClubhouse('/api/clubhouse/fixtures', {
                match_date: this.fixtureDraft.match_date,
                kickoff: this.fixtureDraft.kickoff,
                home_team: this.fixtureDraft.home ? club : opponent,
                away_team: this.fixtureDraft.home ? opponent : club,
                venue: this.fixtureDraft.venue.trim(),
            }, 'POST', 'Fixture added to the schedule.');
            if (!this.clubhouseMessage.startsWith('Fixture added')) return;
            this.fixtureDraft = { match_date: '', kickoff: '', opponent: '', venue: '', home: true };
        },
        async submitLeagueResult() {
            await this.writeClubhouse('/api/clubhouse/results', {
                ...this.resultDraft,
                reported_by: this.resultDraft.reported_by.trim(),
                venue: this.resultDraft.venue.trim(),
            }, 'POST', 'Result saved and the league table recalculated.');
            if (!this.clubhouseMessage.startsWith('Result saved')) return;
            this.resultDraft = { match_date: '', home_team: '', away_team: '', home_score: '', away_score: '', venue: '', reported_by: this.clubName };
        },
        async writeClubhouse(url, payload, method, successMessage, allowWithoutKey = false) {
            if (!this.clubhouseKeyVerified && !(allowWithoutKey && !this.clubhouse.uploads_enabled)) {
                this.clubhouseMessage = 'Verify the shared clubhouse key in Club setup first.';
                return;
            }
            this.clubhouseBusy = true;
            this.clubhouseMessage = '';
            try {
                await this.requestJson(url, {
                    method,
                    headers: { 'Content-Type': 'application/json', 'X-Clubhouse-Key': this.clubhouseKey },
                    body: JSON.stringify(payload),
                });
                await this.loadClubhouse();
                this.clubhouseMessage = successMessage;
            } catch (error) {
                this.clubhouseMessage = error.message;
                if (/key/i.test(error.message)) this.clubhouseKeyVerified = false;
            } finally {
                this.clubhouseBusy = false;
            }
        },
        async loadClubhouse() {
            this.clubhouse = await this.requestJson('/api/clubhouse');
            this.clubNameDraft = this.clubhouse.club_name;
            this.resultDraft.reported_by ||= this.clubhouse.club_name;
        },
        async switchDataSource(source) {
            if (this.dataSource === source) return;
            this.dataSource = source;
            localStorage.setItem('dashboard-data-source', source);
            this.selectedMatchId = null;
            await this.loadWorkspace();
        },
        normalizeTeam(team) {
            return String(team || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
        },
        clubTeamStats(report) {
            const rows = report?.team_stats || [];
            return rows.find((row) => this.normalizeTeam(row.team) === this.normalizeTeam(this.clubTeam)) || rows[0] || null;
        },
        resultFor(report) {
            const rows = report?.team_stats || [];
            if (rows.length < 2) return null;
            const club = this.clubTeamStats(report);
            const opponent = rows.find((row) => row !== club);
            if (!club || !opponent) return null;
            const score = (row) => Number(row.n_goals) || 0;
            const forGoals = score(club);
            const againstGoals = score(opponent);
            return {
                for: forGoals,
                against: againstGoals,
                opponent: opponent.team,
                outcome: forGoals > againstGoals ? 'W' : forGoals < againstGoals ? 'L' : 'D',
            };
        },
        matchLabel(match) {
            return { opponent: match?.label || 'Match', homeAway: '' };
        },
        changePage(page) {
            this.page = page;
            this.mobileNavOpen = false;
            this.accountOpen = false;
        },
        openMatch(match) {
            if (!match) return;
            this.selectedMatchId = match.id;
            this.page = 'report';
            this.mobileNavOpen = false;
        },
        openRosterEditor() {
            this.rosterDraft = this.squad.map((player) => ({
                track_id: player.track_id,
                jersey_number: player.jersey_number,
                name: player.name,
                position: player.position,
                photo_url: player.photo_url || '',
            }));
            this.rosterEditorOpen = true;
        },
        async saveRoster() {
            this.rosterSaving = true;
            try {
                await this.requestJson('/api/roster', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.rosterDraft),
                });
                this.rosterEditorOpen = false;
                await this.loadWorkspace();
            } catch (error) {
                this.loadError = error.message;
            } finally {
                this.rosterSaving = false;
            }
        },
        toggleReportColumn(key) {
            const column = this.reportColumns.find((item) => item.key === key);
            if (column) column.active = !column.active;
        },
        reportColumnActive(key) {
            return this.reportColumns.some((column) => column.key === key && column.active);
        },
        toggleSquadColumn(key) {
            const column = this.squadColumns.find((item) => item.key === key);
            if (column) column.active = !column.active;
        },
        squadColumnActive(key) {
            return this.squadColumns.some((column) => column.key === key && column.active);
        },
        portraitFallback(player) {
            const number = String(player?.jersey_number || player?.track_id || '?');
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 280"><defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#315a9c"/><stop offset="1" stop-color="#142640"/></linearGradient></defs><rect width="240" height="280" fill="url(#bg)"/><circle cx="120" cy="84" r="42" fill="#d7a27c"/><path d="M77 82c5-47 85-60 91 0-18-17-59-18-91 0Z" fill="#202738"/><path d="M63 280c4-73 32-103 57-103s53 30 57 103Z" fill="#e84e78"/><text x="120" y="252" fill="#fff" font-size="52" font-family="Arial" font-weight="700" text-anchor="middle">${number}</text></svg>`;
            return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
        },
        usePortraitFallback(event, player) {
            event.target.onerror = null;
            event.target.src = this.portraitFallback(player);
        },
        formatDate(value, options = { day: 'numeric', month: 'short', year: 'numeric' }) {
            if (!value) return 'Date pending';
            const date = new Date(`${value}T00:00:00`);
            return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString('en-GB', options);
        },
        todayDate() {
            const today = new Date();
            today.setMinutes(today.getMinutes() - today.getTimezoneOffset());
            return today.toISOString().slice(0, 10);
        },
        formatNumber(value, digits = 0) {
            const number = Number(value);
            return Number.isFinite(number) ? number.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '--';
        },
        metricFor(report, key) {
            const row = this.clubTeamStats(report);
            const value = Number(row?.[key]);
            return Number.isFinite(value) ? value : null;
        },
        percentValue(value) {
            const number = Number(value);
            if (!Number.isFinite(number)) return null;
            return Math.abs(number) <= 1 ? number * 100 : number;
        },
        openSignIn() {
            this.signInName = this.accountName;
            this.signInEmail = this.accountEmail;
            this.signInRole = this.role;
            this.signInOpen = true;
            this.accountOpen = false;
        },
        saveAccount() {
            this.role = this.signInRole;
            this.accountName = this.signInName.trim();
            this.accountEmail = this.signInEmail.trim();
            localStorage.setItem('club-role', this.role);
            localStorage.setItem('club-name', this.accountName);
            localStorage.setItem('club-email', this.accountEmail);
            this.signInOpen = false;
            this.changePage(this.role === 'player' ? 'squad' : 'overview');
        },
        async submitQuestion(question = this.assistantQuestion) {
            const text = String(question || '').trim();
            if (!text || this.assistantBusy) return;
            this.assistantMessages.push({ role: 'user', text });
            this.assistantQuestion = '';
            this.assistantBusy = true;
            await this.$nextTick();
            const answer = this.answerQuestion(text);
            this.assistantMessages.push({ role: 'assistant', text: answer });
            this.assistantBusy = false;
        },
        answerQuestion(question) {
            const text = question.toLowerCase();
            const knownResults = this.seasonResults.filter((match) => match.result);
            if (/down|behind|comeback|trailing/.test(text)) {
                const comebackWins = knownResults.filter((match) => {
                    const goals = match.report.goals || [];
                    if (!goals.length) return false;
                    let ourScore = 0;
                    let theirScore = 0;
                    let trailed = false;
                    goals.slice().sort((a, b) => Number(a.timestamp_s) - Number(b.timestamp_s)).forEach((goal) => {
                        if (this.normalizeTeam(goal.team) === this.normalizeTeam(this.clubTeam)) ourScore += 1;
                        else theirScore += 1;
                        if (theirScore > ourScore) trailed = true;
                    });
                    return trailed && match.result.outcome === 'W';
                });
                const winsWithTimeline = knownResults.filter((match) => match.result.outcome === 'W' && match.report.goals?.length);
                if (!winsWithTimeline.length) return 'I cannot verify comebacks from the imported reports yet. This requires goal events with timestamps and team attribution.';
                return `${comebackWins.length} of ${winsWithTimeline.length} wins (${Math.round(comebackWins.length / winsWithTimeline.length * 100)}%) came after the team had been behind. I used timestamped goal events; wins without a complete event timeline are excluded.`;
            }
            if (/distance|ran|running|run|result|correlat|related/.test(text)) {
                const pairs = knownResults.map((match) => {
                    const distance = this.metricFor(match.report, 'total_distance_km');
                    if (distance === null) return null;
                    const points = match.result.outcome === 'W' ? 3 : match.result.outcome === 'D' ? 1 : 0;
                    return { distance, points };
                }).filter(Boolean);
                if (pairs.length < 3) return `There are ${pairs.length} scored matches with distance data. I need at least 3 to estimate whether distance and results move together; more matches will make the comparison more useful.`;
                const xMean = pairs.reduce((sum, item) => sum + item.distance, 0) / pairs.length;
                const yMean = pairs.reduce((sum, item) => sum + item.points, 0) / pairs.length;
                const numerator = pairs.reduce((sum, item) => sum + (item.distance - xMean) * (item.points - yMean), 0);
                const xVariance = pairs.reduce((sum, item) => sum + (item.distance - xMean) ** 2, 0);
                const yVariance = pairs.reduce((sum, item) => sum + (item.points - yMean) ** 2, 0);
                if (!xVariance || !yVariance) return `Distance and points are available for ${pairs.length} matches, but there is not enough variation to calculate a correlation.`;
                const correlation = numerator / Math.sqrt(xVariance * yVariance);
                const direction = Math.abs(correlation) < 0.2 ? 'little linear relationship' : correlation > 0 ? 'a positive relationship' : 'a negative relationship';
                return `Across ${pairs.length} matches, distance covered and league points show ${direction} (Pearson r = ${correlation.toFixed(2)}). This is a small-sample association, not evidence that running more causes a result.`;
            }
            if (/win|record|season|result|form/.test(text)) {
                const known = this.wins + this.draws + this.losses;
                return known ? `The current record is ${this.wins} wins, ${this.draws} draws and ${this.losses} losses from ${known} matches with recorded team scores (${this.winRate}% wins).` : 'I do not have enough recorded team scores to summarize results yet. Add match reports with team goals to build the season record.';
            }
            if (/goal|scor|attack/.test(text)) return `The team has ${this.seasonTotals.goals} recorded goals across ${this.seasonTotals.games} match reports. Ask about a specific match to see its event timeline and player contributions.`;
            if (/pass|possession/.test(text)) {
                const possession = this.seasonTotals.averagePossession;
                return `The team has ${this.seasonTotals.passes.toLocaleString('en-GB')} recorded passes${possession === null ? '' : ` and averaged ${possession.toFixed(1)}% possession per report`} this season.`;
            }
            return 'I can answer questions about results, comebacks, distance covered, goals, passes and possession using the imported match reports. Try one of the suggested questions below.';
        },
        assistantGreeting() {
            return 'I can help you read the season. Ask about results, player output, distance or possession.';
        },
        formatOutcome(outcome) {
            return ({ W: 'Win', D: 'Draw', L: 'Loss' })[outcome] || 'Report ready';
        },
    },
    mounted() {
        this.assistantMessages = [{ role: 'assistant', text: this.assistantGreeting() }];
        if (this.clubhouseKey) this.verifyClubhouseKey();
        this.loadWorkspace();
    },
}).mount('#app');