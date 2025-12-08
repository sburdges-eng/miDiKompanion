// ContentLibrary - Library management system

export interface LibraryItem {
  id: string;
  name: string;
  type: 'loop' | 'one-shot' | 'instrument' | 'preset' | 'effect-preset' | 'sound-pack';
  path: string;
  tags: string[];
  rating: number; // 0-5
  color: string;
  metadata: {
    bpm?: number;
    key?: string;
    genre?: string;
    duration?: number;
    sampleRate?: number;
    bitDepth?: number;
    format?: string;
  };
  license?: string;
  cloudSynced: boolean;
  downloaded: boolean;
}

export interface SmartCollection {
  id: string;
  name: string;
  query: {
    type?: string[];
    tags?: string[];
    minRating?: number;
    bpm?: { min?: number; max?: number };
    key?: string;
    genre?: string;
  };
}

export class ContentLibrary {
  private items: Map<string, LibraryItem> = new Map();
  private collections: Map<string, SmartCollection> = new Map();
  private tags: Set<string> = new Set();
  private downloadQueue: string[] = [];
  private cloudSyncEnabled: boolean = false;

  // 925. Factory content library
  loadFactoryContent(): LibraryItem[] {
    // Load factory content
    return Array.from(this.items.values()).filter((item) => !item.path.includes('/user/'));
  }

  // 926. Sound packs
  addSoundPack(_packId: string, items: LibraryItem[]): void {
    items.forEach((item) => {
      item.type = 'sound-pack';
      this.items.set(item.id, item);
    });
  }

  // 927. Loop libraries
  getLoops(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.type === 'loop');
  }

  // 928. One-shot libraries
  getOneShots(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.type === 'one-shot');
  }

  // 929. Instrument libraries
  getInstruments(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.type === 'instrument');
  }

  // 930. Preset libraries
  getPresets(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.type === 'preset');
  }

  // 931. Effect preset libraries
  getEffectPresets(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.type === 'effect-preset');
  }

  // 932. Download manager
  addToDownloadQueue(itemId: string): void {
    if (!this.downloadQueue.includes(itemId)) {
      this.downloadQueue.push(itemId);
    }
  }

  getDownloadQueue(): string[] {
    return [...this.downloadQueue];
  }

  async downloadItem(itemId: string): Promise<void> {
    const item = this.items.get(itemId);
    if (!item) return;

    // Simulate download
    item.downloaded = true;
    this.downloadQueue = this.downloadQueue.filter((id) => id !== itemId);
  }

  // 933. License management
  setLicense(itemId: string, license: string): void {
    const item = this.items.get(itemId);
    if (item) {
      item.license = license;
    }
  }

  getLicense(itemId: string): string | undefined {
    return this.items.get(itemId)?.license;
  }

  // 934. Cloud library sync
  enableCloudSync(enabled: boolean): void {
    this.cloudSyncEnabled = enabled;
  }

  async syncToCloud(): Promise<void> {
    if (!this.cloudSyncEnabled) return;

    const itemsToSync = Array.from(this.items.values()).filter((item) => !item.cloudSynced);
    // Simulate cloud sync
    itemsToSync.forEach((item) => {
      item.cloudSynced = true;
    });
  }

  // 935. User library
  addUserItem(item: LibraryItem): void {
    this.items.set(item.id, item);
    this.updateTags(item.tags);
  }

  getUserItems(): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.path.includes('/user/'));
  }

  // 936. Sample organization
  organizeByTags(): Map<string, LibraryItem[]> {
    const organized = new Map<string, LibraryItem[]>();
    this.items.forEach((item) => {
      item.tags.forEach((tag) => {
        if (!organized.has(tag)) {
          organized.set(tag, []);
        }
        organized.get(tag)!.push(item);
      });
    });
    return organized;
  }

  // 937. Tagging system
  addTag(itemId: string, tag: string): void {
    const item = this.items.get(itemId);
    if (item) {
      if (!item.tags.includes(tag)) {
        item.tags.push(tag);
        this.updateTags([tag]);
      }
    }
  }

  removeTag(itemId: string, tag: string): void {
    const item = this.items.get(itemId);
    if (item) {
      item.tags = item.tags.filter((t) => t !== tag);
    }
  }

  getAllTags(): string[] {
    return Array.from(this.tags);
  }

  // 938. Rating system
  setRating(itemId: string, rating: number): void {
    const item = this.items.get(itemId);
    if (item) {
      item.rating = Math.max(0, Math.min(5, rating));
    }
  }

  getItemsByRating(minRating: number): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.rating >= minRating);
  }

  // 939. Color coding
  setColor(itemId: string, color: string): void {
    const item = this.items.get(itemId);
    if (item) {
      item.color = color;
    }
  }

  getItemsByColor(color: string): LibraryItem[] {
    return Array.from(this.items.values()).filter((item) => item.color === color);
  }

  // 940. Smart collections
  createSmartCollection(collection: SmartCollection): void {
    this.collections.set(collection.id, collection);
  }

  getSmartCollection(id: string): LibraryItem[] {
    const collection = this.collections.get(id);
    if (!collection) return [];

    return this.search(collection.query);
  }

  // 941. Database search
  search(query: {
    type?: string[];
    tags?: string[];
    minRating?: number;
    bpm?: { min?: number; max?: number };
    key?: string;
    genre?: string;
    name?: string;
  }): LibraryItem[] {
    let results = Array.from(this.items.values());

    if (query.type && query.type.length > 0) {
      results = results.filter((item) => query.type!.includes(item.type));
    }

    if (query.tags && query.tags.length > 0) {
      results = results.filter((item) =>
        query.tags!.some((tag) => item.tags.includes(tag))
      );
    }

    if (query.minRating !== undefined) {
      results = results.filter((item) => item.rating >= query.minRating!);
    }

    if (query.bpm) {
      results = results.filter((item) => {
        const itemBpm = item.metadata.bpm;
        if (!itemBpm) return false;
        if (query.bpm!.min && itemBpm < query.bpm!.min) return false;
        if (query.bpm!.max && itemBpm > query.bpm!.max) return false;
        return true;
      });
    }

    if (query.key) {
      results = results.filter((item) => item.metadata.key === query.key);
    }

    if (query.genre) {
      results = results.filter((item) => item.metadata.genre === query.genre);
    }

    if (query.name) {
      const nameLower = query.name.toLowerCase();
      results = results.filter((item) => item.name.toLowerCase().includes(nameLower));
    }

    return results;
  }

  private updateTags(newTags: string[]): void {
    newTags.forEach((tag) => this.tags.add(tag));
  }

  // Getters
  getItem(itemId: string): LibraryItem | undefined {
    return this.items.get(itemId);
  }

  getAllItems(): LibraryItem[] {
    return Array.from(this.items.values());
  }

  getCollections(): SmartCollection[] {
    return Array.from(this.collections.values());
  }
}
