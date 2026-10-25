/*
 * Copyright 2026 Dukehoff GmbH
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dukehoff.decide.showcase;

import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;

/** The typed decision for one utterance of the MASSIVE dataset: which part of a voice assistant handles it. */
@Question("You route requests that users say to a voice assistant to the right part of the assistant.")
public record AssistantRequest(@Question("Which part of the assistant should handle the request?") Scenario scenario) {

    public enum Scenario {
        @Option("Setting, checking or removing alarms") ALARM,
        @Option("Volume: louder, quieter, mute") AUDIO,
        @Option("Appointments, meetings and reminders in the calendar") CALENDAR,
        @Option("Recipes and cooking questions") COOKING,
        @Option("Current date or time, time zones") DATETIME,
        @Option("Reading or sending email, email contacts") EMAIL,
        @Option("Small talk: greetings, jokes, odd remarks") GENERAL,
        @Option("Smart home devices: lights, plugs, coffee machine, vacuum cleaner") IOT,
        @Option("To-do and shopping lists") LISTS,
        @Option("Information about music being played, likes and dislikes, playback settings") MUSIC,
        @Option("News") NEWS,
        @Option("Start playing music, radio, podcasts, audiobooks or games") PLAY,
        @Option("General knowledge questions, definitions, currency, stocks, maths") QA,
        @Option("Suggestions for events, places or movies") RECOMMENDATION,
        @Option("Posting to or reading social media") SOCIAL,
        @Option("Ordering or asking about takeaway food") TAKEAWAY,
        @Option("Trains, tickets, taxis and traffic") TRANSPORT,
        @Option("Weather") WEATHER
    }
}
