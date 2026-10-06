const { createApp } = Vue;

createApp({
    data() {
        return {
            matches: [],
            reports: {},
            selectedMatchId: null,
            reportMapPlayerId: '',
            selectedShot: null,
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
        reportEvents() {
            return this.currentReport?.event_rows || [];
        },
        reportShots() {
            return this.reportEvents.filter((event) => event.event_type === 'shot' && this.hasPitchLocation(event));
        },
        selectedShotDetails() {
            if (!this.selectedShot) return null;
            const shot = this.selectedShot;
            const player = (this.currentReport?.players || []).find((item) => String(item.track_id) === String(shot.player_track_id));
            return {
                player: player?.name || (shot.player_track_id == null ? 'Player not identified' : `Player ${shot.player_track_id}`),
                team: this.normalizeTeam(shot.team) === this.normalizeTeam(this.clubTeam) ? this.clubName : (shot.team || 'Opponent'),
                minute: Math.floor((Number(shot.timestamp_s) || 0) / 60),
                outcome: shot.outcome || 'Outcome unavailable',
                xg: Number.isFinite(Number(shot.xg)) ? Number(shot.xg) : null,
            };
        },
        reportMapPlayers() {
            const ids = new Set(this.reportEvents
                .filter((event) => event.player_track_id !== null && event.player_track_id !== undefined && this.hasPitchLocation(event))
                .map((event) => String(event.player_track_id)));
            return (this.currentReport?.players || []).filter((player) => ids.has(String(player.track_id)));
        },
        reportMapEvents() {
            const selectedId = String(this.reportMapPlayerId || this.reportMapPlayers[0]?.track_id || '');
            if (!selectedId) return [];
            return this.reportEvents.filter((event) =>
                String(event.player_track_id) === selectedId
                && this.hasPitchLocation(event)
                && ['shot', 'tackle', 'pass', 'interception', 'recovery', 'carry'].includes(event.event_type)
            );
        },
        reportComparisons() {
            const teams = this.currentReport?.team_stats || [];
            if (teams.length < 2) return [];
            const metrics = [
                { key: 'possession_pct', label: 'Possession', suffix: '%', percent: true, digits: 1 },
                { key: 'n_shots', label: 'Shots', suffix: '', digits: 0 },
                { key: 'shots_on_target', label: 'On target', suffix: '', digits: 0 },
                { key: 'n_passes', label: 'Passes completed', suffix: '', digits: 0 },
                { key: 'tackles', label: 'Tackles', suffix: '', digits: 0, eventType: 'tackle' },
                { key: 'interceptions', label: 'Interceptions', suffix: '', digits: 0, eventType: 'interception' },
                { key: 'throw_in', label: 'Throw-ins', suffix: '', digits: 0, eventType: 'throw_in' },
                { key: 'offside', label: 'Offsides', suffix: '', digits: 0, eventType: 'offside' },
                { key: 'total_distance_km', label: 'Distance', suffix: ' km', digits: 1 },
                { key: 'xg', label: 'Expected goals', suffix: ' xG', digits: 2 },
            ];
            return metrics.map((metric) => {
                const values = teams.map((team) => {
                    const raw = metric.eventType
                        ? (this.eventCoverage(metric.eventType) === 'unavailable' ? null : this.teamEventCount(team.team, metric.eventType))
                        : team[metric.key];
                    const value = raw === null || raw === undefined || raw === '' ? null : Number(raw);
                    return { team, value: Number.isFinite(value) ? (metric.percent ? this.percentValue(value) : value) : null };
                });
                const max = Math.max(1, ...values.map((item) => item.value ?? 0));
                return {
                    ...metric,
                    values: values.map((item) => ({
                        team: item.team,
                        name: this.normalizeTeam(item.team.team) === this.normalizeTeam(this.clubTeam) ? this.clubName : item.team.team,
                        value: item.value,
                        display: item.value === null ? 'Not available' : `${this.formatNumber(item.value, metric.digits)}${metric.suffix}`,
                        width: item.value === null ? 0 : Math.max(4, item.value / max * 100),
                    })),
                };
            });
        },
        reportFlowBins() {
            const bins = Array.from({ length: 9 }, (_, index) => ({
                start: index * 10,
                end: (index + 1) * 10,
                home: 0,
                away: 0,
            }));
            this.reportEvents.filter((event) => ['shot', 'goal', 'tackle'].includes(event.event_type)).forEach((event) => {
                const minute = Math.max(0, Number(event.timestamp_s) || 0) / 60;
                const bin = bins[Math.min(8, Math.floor(minute / 10))];
                if (this.normalizeTeam(event.team) === this.normalizeTeam(this.clubTeam)) bin.home += 1;
                else bin.away += 1;
            });
            const peak = Math.max(1, ...bins.map((bin) => Math.max(bin.home, bin.away)));
            return bins.map((bin) => ({
                ...bin,
                homeHeight: bin.home ? Math.max(4, bin.home / peak * 100) : 0,
                awayHeight: bin.away ? Math.max(4, bin.away / peak * 100) : 0,
            }));
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
            this.selectedShot = null;
            const firstLocatedPlayer = (match.report?.event_rows || []).find((event) =>
                event.player_track_id !== null && event.player_track_id !== undefined && this.hasPitchLocation(event)
            );
            this.reportMapPlayerId = firstLocatedPlayer ? String(firstLocatedPlayer.player_track_id) : '';
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
            const rawValue = row?.[key];
            if (rawValue === null || rawValue === undefined || rawValue === '') return null;
            const value = Number(rawValue);
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
            const answer = await this.answerQuestion(text);
            this.assistantMessages.push({ role: 'assistant', text: answer });
            this.assistantBusy = false;
        },
        answerQuestion(question) {
            const normalize = (value) => String(value || '').toLocaleLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
            const text = normalize(question);
            const requestedMatch = this.sortedMatches.find((match) => {
                const label = normalize(match.label);
                if (!label) return false;
                if (text.includes(label)) return true;
                const opponent = match.report?.team_stats?.find((team) => normalize(team.team) !== normalize(this.clubTeam))?.team;
                const opponentName = normalize(opponent);
                if (opponentName && text.includes(opponentName)) return true;
                const opponentWords = opponentName?.split(/[^a-z0-9]+/).filter((word) => word.length > 3) || [];
                if (opponentWords.some((word) => text.includes(word))) return true;
                const identifyingWords = label.split(/[^a-z0-9]+/).filter((word) => word.length > 3);
                return identifyingWords.length > 0 && identifyingWords.filter((word) => text.includes(word)).length >= Math.min(2, identifyingWords.length);
            });
            const requestedPlayer = this.squad.find((player) => {
                const name = normalize(player.name);
                return name.length > 2 && text.includes(name);
            });
            const teamNames = [...new Set(this.seasonResults.flatMap((match) => match.report?.team_stats?.map((team) => team.team) || []))];
            const requestedTeam = teamNames.find((team) => {
                const name = normalize(team);
                if (!name || name === normalize(this.clubTeam)) return false;
                if (text.includes(name)) return true;
                const words = name.split(/[^a-z0-9]+/).filter((word) => word.length > 3);
                return words.length > 0 && words.every((word) => text.includes(word));
            });
            if (/best player|highest rated|top rated/.test(text)) {
                const ratedPlayers = this.squad
                    .map((player) => ({ player, rating: Number(player.rating) }))
                    .filter((item) => Number.isFinite(item.rating))
                    .sort((a, b) => b.rating - a.rating)
                    .slice(0, 3);
                if (ratedPlayers.length) {
                    return `Highest-rated players in the loaded reports: ${ratedPlayers.map((item) => `${item.player.name} (${this.formatNumber(item.rating, 1)})`).join(', ')}.`;
                }
                return 'Player ratings are not available in the loaded reports.';
            }
            const scopes = requestedMatch
                ? [{ label: requestedMatch.label, report: requestedMatch.report, result: requestedMatch.result }]
                : this.seasonResults.map((match) => ({ label: match.label, report: match.report, result: match.result }));
            const metrics = [
                { key: 'goals', label: 'goals', aliases: ['goals', 'goal', 'scored'], getter: (player, team) => player?.goals ?? player?.n_goals ?? team?.n_goals },
                { key: 'assists', label: 'assists', aliases: ['assists', 'assist'], getter: (player, team) => player?.assists ?? team?.assists },
                { key: 'xg', label: 'expected goals (xG)', aliases: ['expected goals', 'expected goal', 'xg'], getter: (player, team) => player?.xg ?? team?.xg },
                { key: 'shots_on_target', label: 'shots on target', aliases: ['shots on target', 'on target'], getter: (player, team) => player?.shots_on_target ?? team?.shots_on_target },
                { key: 'shots', label: 'shots', aliases: ['shots', 'shot', 'shooting'], getter: (player, team) => player?.n_shots ?? team?.n_shots },
                { key: 'passes', label: 'completed passes', aliases: ['completed passes', 'passes', 'pass'], getter: (player, team) => player?.passes_completed ?? player?.n_passes ?? team?.passes_completed ?? team?.n_passes },
                { key: 'pass_accuracy', label: 'pass completion', aliases: ['pass completion', 'pass accuracy', 'completion'], getter: (player, team) => {
                    const rate = player?.pass_completion_rate ?? team?.pass_completion_rate;
                    return rate == null ? null : Number(rate) * 100;
                } },
                { key: 'tackles', label: 'tackles', aliases: ['tackles', 'tackle'], getter: (player, team) => player?.tackles ?? team?.tackles },
                { key: 'interceptions', label: 'interceptions', aliases: ['interceptions', 'interception'], eventType: 'interception', getter: (player, team) => player?.interceptions ?? team?.interceptions },
                { key: 'distance', label: 'distance covered', aliases: ['distance', 'running', 'ran', 'kilometres', 'kilometers'], getter: (player, team) => player ? Number(player.distance_m) / 1000 : team?.total_distance_km },
                { key: 'speed', label: 'top speed', aliases: ['top speed', 'fastest', 'speed'], getter: (player) => player?.top_speed_kmh },
                { key: 'rating', label: 'rating', aliases: ['rating', 'rated'], getter: (player) => player?.rating },
                { key: 'possession', label: 'possession', aliases: ['possession', 'ball control'], getter: (player, team) => this.percentValue(player?.possession_pct ?? team?.possession_pct) },
                { key: 'sprints', label: 'sprints', aliases: ['sprints', 'sprint'], getter: (player) => player?.n_sprints },
                { key: 'carries', label: 'carries', aliases: ['carries', 'carry', 'dribbles', 'dribble'], getter: (player) => player?.carries },
                { key: 'throw_ins', label: 'throw-ins', aliases: ['throw-ins', 'throw ins', 'throwin', 'throwins'], eventType: 'throw_in' },
                { key: 'offsides', label: 'offsides', aliases: ['offsides', 'offside'], eventType: 'offside' },
            ];
            const requestedMetrics = metrics.filter((metric) => metric.aliases.some((alias) => text.includes(alias)));
            const facts = [];
            for (const metric of requestedMetrics) {
                const rows = [];
                let unavailable = false;
                for (const scope of scopes) {
                    const report = scope.report;
                    if (!report) continue;
                    const player = requestedPlayer
                        ? (report.players || []).find((item) => String(item.track_id) === String(requestedPlayer.track_id))
                        : null;
                    if (requestedPlayer && !player) continue;
                    const team = requestedTeam
                        ? report.team_stats?.find((item) => normalize(item.team) === normalize(requestedTeam))
                        : this.clubTeamStats(report);
                    if (metric.eventType) {
                        if (!report.demo && !(report.event_capabilities || []).includes(metric.eventType)) {
                            unavailable = true;
                            continue;
                        }
                        const count = (report.event_rows || []).filter((event) =>
                            (metric.eventType === 'interception'
                                ? event.event_type === 'interception' || event.outcome === 'intercepted'
                                : event.event_type === metric.eventType)
                            && this.normalizeTeam(event.team) === this.normalizeTeam(requestedTeam || this.clubTeam)
                            && (!player || String(event.player_track_id) === String(player.track_id))
                        ).length;
                        rows.push({ label: scope.label, value: count });
                        continue;
                    }
                    const rawValue = metric.getter(player, team);
                    const value = rawValue === null || rawValue === undefined || rawValue === '' ? null : Number(rawValue);
                    if (Number.isFinite(value)) rows.push({ label: scope.label, value });
                }
                if (!rows.length) {
                    const subject = requestedPlayer ? `${requestedPlayer.name}: ` : '';
                    facts.push(unavailable
                        ? `${subject}${metric.label} are not tracked in the loaded real reports.`
                        : requestedPlayer
                            ? `${subject}there is no ${metric.label} data in these reports.`
                            : `${metric.label} data is not available in the loaded reports.`);
                    continue;
                }
                const total = rows.reduce((sum, row) => sum + row.value, 0);
                const average = total / rows.length;
                const digits = ['xg', 'distance', 'speed', 'rating', 'possession', 'pass_accuracy'].includes(metric.key) ? 1 : 0;
                const unit = metric.key === 'distance' ? ' km' : metric.key === 'speed' ? ' km/h' : ['possession', 'pass_accuracy'].includes(metric.key) ? '%' : '';
                const averageAsked = /average|avg|per match|per game|usually|typically/.test(text);
                const subject = requestedPlayer?.name || requestedTeam || this.clubName;
                const scopeText = requestedMatch ? `in ${requestedMatch.label}` : `across ${rows.length} report${rows.length === 1 ? '' : 's'}`;
                facts.push(`${subject}: ${metric.label} ${scopeText}: ${this.formatNumber(averageAsked ? average : total, digits)}${unit}${averageAsked ? ' on average' : requestedMatch ? '' : ' total'}.`);
            }
            if (/who|which player|top|most|least|best|highest|lowest|leader/.test(text)) {
                const rankMetric = metrics.find((metric) => metric.aliases.some((alias) => text.includes(alias)))
                    || (text.includes('best') || text.includes('leader') ? metrics.find((metric) => metric.key === 'rating') : null);
                if (rankMetric) {
                    const ranking = this.squad.map((player) => {
                        const raw = rankMetric.getter(player, null);
                        const value = raw === null || raw === undefined || raw === '' ? NaN : Number(raw);
                        return { player, value };
                    }).filter((item) => Number.isFinite(item.value));
                    if (ranking.length) {
                        ranking.sort((a, b) => b.value - a.value);
                        const leaders = ranking.slice(0, 3).map((item) => `${item.player.name} (${this.formatNumber(item.value, ['xg', 'distance', 'speed', 'rating'].includes(rankMetric.key) ? 1 : 0)})`);
                        return `Top ${rankMetric.label} in the loaded squad reports: ${leaders.join(', ')}. This is calculated from ${this.isDemo ? 'sample' : 'available'} match data.`;
                    }
                }
            }
            if (facts.length) return facts.join(' ');

            if (requestedMatch) {
                const result = requestedMatch.result;
                const team = this.clubTeamStats(requestedMatch.report);
                const opponent = requestedMatch.report?.team_stats?.find((row) => this.normalizeTeam(row.team) !== this.normalizeTeam(this.clubTeam));
                if (!result || !team) return `I found ${requestedMatch.label}, but its report does not contain enough team data to answer in detail.`;
                return `${requestedMatch.label}: ${result.for}–${result.against} (${this.formatOutcome(result.outcome)}). ${this.formatNumber(this.percentValue(team.possession_pct), 1)}% possession, ${this.formatNumber(team.n_shots)} shots, ${team.xg == null ? 'xG unavailable' : `${this.formatNumber(team.xg, 2)} xG`}, and ${this.formatNumber(team.n_passes)} completed passes. ${opponent ? `${opponent.team} had ${this.formatNumber(opponent.n_shots)} shots.` : ''}`;
            }
            if (requestedTeam) {
                return `I found ${requestedTeam} in the loaded league reports. Ask about a recorded metric, score or match to see its values.`;
            }
            if (requestedPlayer) {
                const appearances = scopes.filter((scope) => (scope.report?.players || []).some((item) => String(item.track_id) === String(requestedPlayer.track_id)));
                if (!appearances.length) return `${requestedPlayer.name} is in the squad list, but has no player rows in the loaded match reports.`;
                return `${requestedPlayer.name} (${requestedPlayer.position || 'player'}) appears in ${appearances.length} loaded report${appearances.length === 1 ? '' : 's'}. You can ask about any recorded stat, match or event for this player.`;
            }

            const knownResults = this.seasonResults.filter((match) => match.result);
            const matchingMatches = this.sortedMatches.filter((match) => {
                const searchable = normalize(`${match.label} ${match.date}`);
                const tokens = text.split(/[^a-z0-9]+/).filter((token) => token.length > 3);
                return tokens.some((token) => searchable.includes(token));
            }).slice(0, 3);
            const summary = knownResults.length
                ? `${this.clubName}: ${this.wins} wins, ${this.draws} draws and ${this.losses} losses from ${knownResults.length} scored matches, with ${this.seasonTotals.goals} goals and ${this.seasonTotals.passes} completed passes.`
                : `${this.matches.length} match reports are loaded, but reliable scores are not available for a season record.`;
            if (matchingMatches.length) return `${summary} The closest report matches are: ${matchingMatches.map((match) => `${match.label}${match.result ? ` (${match.result.for}–${match.result.against})` : ''}`).join('; ')}.`;
            return `${summary} I searched the loaded match and squad data for “${question}” but could not find a reliable fact for that exact question. I can answer open-ended questions about the club, players, opponents, reports and recorded match events; for unrecorded events or general football knowledge I will say when the data is missing.`;
        },
        assistantGreeting() {
            return 'Ask me anything about the loaded matches, players, opponents, events or team statistics. I will use the available reports and flag missing data.';
        },
        formatOutcome(outcome) {
            return ({ W: 'Win', D: 'Draw', L: 'Loss' })[outcome] || 'Report ready';
        },
        hasPitchLocation(event) {
            const x = Number(event?.location_x);
            const y = Number(event?.location_y);
            return Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 105 && y >= 0 && y <= 68;
        },
        teamEventCount(team, eventType) {
            return this.reportEvents.filter((event) =>
                this.normalizeTeam(event.team) === this.normalizeTeam(team)
                && (eventType === 'interception'
                    ? event.event_type === 'interception' || event.outcome === 'intercepted'
                    : event.event_type === eventType)
            ).length;
        },
        eventCoverage(eventType) {
            if (this.currentReport?.demo) return 'sample';
            return (this.currentReport?.event_capabilities || []).includes(eventType) ? 'tracked' : 'unavailable';
        },
    },
    mounted() {
        this.assistantMessages = [{ role: 'assistant', text: this.assistantGreeting() }];
        if (this.clubhouseKey) this.verifyClubhouseKey();
        this.loadWorkspace();
    },
}).mount('#app');